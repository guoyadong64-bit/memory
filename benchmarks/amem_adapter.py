"""Pinned official A-Mem system with provider/storage initialization adapters.

Official note generation, linking, evolution, enhanced document encoding and
agentic search methods execute unchanged. No benchmark answers enter ingestion.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
from types import SimpleNamespace

from memory_agent.clients import ModelError

REPOSITORY='https://github.com/WujiangXu/A-mem-sys'
REVISION='f303dfc71e07bdc787f4bc135d4cea328ae30e99'
ROOT=Path(__file__).resolve().parents[1]/'runtime'/'baselines'/'a_mem_sys'
JSON_RECOVERY_POLICY = {
    'version': 'amem-json-recovery-v1', 'max_extra_requests': 2,
    'initial_default_max_tokens': 4096, 'recovery_max_tokens': 16384,
    'validation': 'complete JSON object and original official schema; no invented fields',
    'cache': 'retain raw failures; reuse successful responses; distinct recovery requests',
}


class AMemProviderStop(BaseException):
    """Prevent official broad exception handlers from silently degrading a run."""


def source_audit():
    return {'repository':REPOSITORY,'revision':REVISION,'sha256':{
        name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'agentic_memory/memory_system.py','agentic_memory/retrievers.py','agentic_memory/llm_controller.py')}}


class CompletionAdapter:
    def __init__(self,client,path):self.client,self.path=client,path

    @staticmethod
    def _validate(content, response_format):
        import jsonschema
        if not isinstance(content, str):
            raise ValueError('missing_text_content')
        text = content.lstrip('\ufeff').strip()
        fence = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', text, re.S | re.I)
        if fence:
            text = fence.group(1).strip()
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError('expected_json_object')
        if response_format and 'json_schema' in response_format:
            try:
                jsonschema.validate(parsed, response_format['json_schema']['schema'])
            except jsonschema.ValidationError:
                raise ValueError('schema_validation_failed') from None
        return text

    def get_completion(self,prompt,response_format,temperature=1.0,max_tokens=None):
        payload={'model':self.client.settings.llm_model,'temperature':temperature,'max_tokens':max_tokens or 4096,
            'response_format':{'type':'json_object'},**self.client.chat_options(self.client.settings.llm_model),
            'messages':[{'role':'system','content':'You must respond with a JSON object.'},
                        {'role':'user','content':prompt}]}
        key=hashlib.sha256(json.dumps({'request':payload,'schema':response_format},sort_keys=True).encode()).hexdigest()
        cache=self.path/'completions'/f'{key}.json'
        from benchmarks.comparison_worker import atomic_json
        if cache.exists():
            try:
                return self._validate(json.loads(cache.read_text(encoding='utf-8'))['content'], response_format)
            except (ValueError, KeyError, TypeError):
                # Preserve the corrupt accepted cache for diagnosis. The raw
                # response cache below can still supply a previously paid result.
                atomic_json(self.path/'json_recovery'/key/'invalid_accepted_cache.json',
                            {'content': cache.read_text(encoding='utf-8')})
        schema=(response_format or {}).get('json_schema',{}).get('schema',{})
        role='amem_evolution' if 'should_evolve' in schema.get('properties',{}) else 'amem_metadata'
        round_path = self.path/'json_recovery'/key/'retry_round.json'
        recovery_round = json.loads(round_path.read_text(encoding='utf-8'))['round'] if round_path.exists() else 0
        request = dict(payload)
        error_kind = 'unknown'
        for attempt in range(JSON_RECOVERY_POLICY['max_extra_requests'] + 1):
            try:
                response=self.client.post(self.client.settings.llm_url.rstrip('/')+'/chat/completions',
                                          self.client.settings.llm_key,request,role)
            except ModelError as exc:
                raise AMemProviderStop('A-Mem provider request failed: '+str(exc)) from None
            except Exception as exc:
                raise AMemProviderStop('A-Mem provider request failed: '+type(exc).__name__) from None
            try:
                choice=response['choices'][0]
                if choice.get('finish_reason') == 'length':
                    raise ValueError('output_truncated')
                content=self._validate(choice['message']['content'], response_format)
            except Exception as exc:
                truncated = (isinstance(response, dict) and bool(response.get('choices'))
                             and response['choices'][0].get('finish_reason') == 'length')
                error_kind = 'output_truncated' if truncated else type(exc).__name__
                atomic_json(self.path/'json_recovery'/key/f'round_{recovery_round}_attempt_{attempt}.json', {
                    'role': role, 'attempt': attempt, 'round': recovery_round, 'error': error_kind,
                    'max_tokens': request['max_tokens'], 'response': response})
                if attempt == JSON_RECOVERY_POLICY['max_extra_requests']:
                    break
                request = dict(payload)
                if truncated:
                    request['max_tokens'] = min(
                        max(payload['max_tokens'], 4096) * (2 ** (attempt + 1)),
                        max(payload['max_tokens'], JSON_RECOVERY_POLICY['recovery_max_tokens']))
                else:
                    # Feedback is format-only and contains no task labels or
                    # generated replacement memory. Changing the request key
                    # avoids replaying the same malformed cached response.
                    request['max_tokens'] = max(payload['max_tokens'], 8192)
                    request['messages'] = [*payload['messages'], {'role': 'user', 'content':
                        f'JSON format retry {attempt + 1}. Return only a complete JSON object '
                        'matching the original schema. Keep string values concise. Do not use '
                        'Markdown fences or explanatory text. Schema: '+json.dumps(schema)}]
                if recovery_round:
                    # If all prior repairs were invalid, a manually resumed run
                    # must not be permanently stuck replaying their cache keys.
                    request['messages'] = [*request['messages'], {'role': 'user', 'content':
                        f'JSON recovery cycle {recovery_round + 1}, attempt {attempt + 1}. '
                        'Return complete valid JSON for the original request; keep values concise.'}]
                print(f'[A-Mem JSON恢复] {role} {error_kind}；重试 {attempt + 1}/2，'
                      f'输出上限 {request["max_tokens"]} token', flush=True)
                continue
            atomic_json(cache, {'content': content})
            return content
        atomic_json(round_path, {'round': recovery_round + 1})
        raise AMemProviderStop('A-Mem structured output invalid after bounded recovery: '+error_kind)


def create_memory(path,client):
    """Keep all official methods; substitute API client and Chroma initialization."""
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='true'
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    import chromadb
    from chromadb.config import Settings
    from chromadb.api.types import EmbeddingFunction
    import agentic_memory.memory_system as official
    from agentic_memory.retrievers import ChromaRetriever

    class APIEmbedding(EmbeddingFunction):
        def __init__(self):pass
        @staticmethod
        def name():return 'smp_budgeted_api'
        def default_space(self):return 'cosine'  # current official SentenceTransformer EF default
        def get_config(self):return {'model':client.settings.embedding_model}
        @staticmethod
        def build_from_config(config):
            if config.get('model')!=client.settings.embedding_model:raise ValueError('Embedding model mismatch')
            return APIEmbedding()
        def __call__(self,input):
            namespace=(client.settings.embedding_url or client.settings.llm_url).rstrip('/')+'|'+client.settings.embedding_model
            with sqlite3.connect(path/'embedding_cache.sqlite3') as db:
                db.execute('CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)')
                keys=[hashlib.sha256((namespace+'\n'+text).encode()).hexdigest() for text in input]
                saved={key:row[0] for key in keys if (row:=db.execute('SELECT vector FROM embeddings WHERE key=?',(key,)).fetchone())}
                missing=list(dict.fromkeys(key for key in keys if key not in saved))
                if missing:
                    try:vectors=client.embed([input[keys.index(key)] for key in missing])
                    except Exception as exc:raise AMemProviderStop('A-Mem embedding failed: '+type(exc).__name__) from None
                    for key,vector in zip(missing,vectors):
                        saved[key]=json.dumps(vector);db.execute('INSERT INTO embeddings VALUES (?,?)',(key,saved[key]))
                return [json.loads(saved[key]) for key in keys]

    class RuntimeRetriever(ChromaRetriever):
        def __init__(self,collection_name='memories',model_name=None):
            persistent_client=chromadb.PersistentClient(path=str((path/'chroma').resolve()),
                settings=Settings(allow_reset=True,anonymized_telemetry=False))
            class ResumeClient:
                def __getattr__(self,name):return getattr(persistent_client,name)
                def reset(self):
                    # Preserve exactly the persisted official index when resuming.
                    # Neighbor metadata changes are not automatically reindexed
                    # by the official algorithm until its consolidation step.
                    return None
            self.client=ResumeClient() if (path/'notes.json').exists() else persistent_client
            self.embedding_function=APIEmbedding()
            self.collection=self.client.get_or_create_collection(name=collection_name,
                embedding_function=self.embedding_function)

    official.ChromaRetriever=RuntimeRetriever
    official.LLMController=lambda *args,**kwargs:SimpleNamespace(llm=CompletionAdapter(client,path))
    system=official.AgenticMemorySystem(model_name=client.settings.embedding_model,
        llm_backend='openai',llm_model=client.settings.llm_model,evo_threshold=100)
    state_path=path/'notes.json'
    if state_path.exists():
        state=json.loads(state_path.read_text(encoding='utf-8'))
        system.memories={note['id']:official.MemoryNote(**note) for note in state['notes']}
        system.evo_cnt=state['evo_cnt']
        indexed=set(system.retriever.collection.get(include=[])['ids'])
        expected=set(system.memories)
        if expected-indexed:raise ValueError('Persisted A-Mem index is incomplete')
        # A kill between index add and atomic checkpoint may leave an uncommitted
        # note. Roll it back, without rebuilding or refreshing existing neighbors.
        if indexed-expected:system.retriever.collection.delete(ids=list(indexed-expected))
    return system


def save_state(memory,path):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    temp=path/'notes.tmp'
    temp.write_text(json.dumps({'evo_cnt':memory.evo_cnt,'notes':[vars(n) for n in memory.memories.values()]},
                               ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path/'notes.json')


def observation_content(frame):
    return 'Historical robot observation: '+json.dumps({k:frame[k] for k in (
        'frame_index','clock','description','entities','relations','interaction','uncertainty')},ensure_ascii=False)


def evidence(memory,results,provenance):
    """All supported frames come from original note content, not evolved context."""
    result=[]
    for item in results:
        note=memory.read(item['id'])
        if note is None:raise ValueError('Official search returned unknown note')
        result.append({'id':note.id,'content':{'original_content':note.content,
            'context':item.get('context',note.context),'keywords':item.get('keywords',note.keywords),
            'tags':item.get('tags',note.tags),'links':note.links,'timestamp':note.timestamp},
            'frame_indices':provenance[note.id]})
    return result
