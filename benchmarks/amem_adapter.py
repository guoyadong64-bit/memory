"""Pinned official A-Mem system with provider/storage initialization adapters.

Official note generation, linking, evolution, enhanced document encoding and
agentic search methods execute unchanged. No benchmark answers enter ingestion.
"""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

REPOSITORY='https://github.com/WujiangXu/A-mem-sys'
REVISION='f303dfc71e07bdc787f4bc135d4cea328ae30e99'
ROOT=Path(__file__).resolve().parents[1]/'runtime'/'baselines'/'a_mem_sys'


class AMemProviderStop(BaseException):
    """Prevent official broad exception handlers from silently degrading a run."""


def source_audit():
    return {'repository':REPOSITORY,'revision':REVISION,'sha256':{
        name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'agentic_memory/memory_system.py','agentic_memory/retrievers.py','agentic_memory/llm_controller.py')}}


class CompletionAdapter:
    def __init__(self,client,path):self.client,self.path=client,path
    def get_completion(self,prompt,response_format,temperature=1.0,max_tokens=None):
        payload={'model':self.client.settings.llm_model,'temperature':temperature,'max_tokens':max_tokens or 4096,
            'response_format':{'type':'json_object'},**self.client.chat_options(self.client.settings.llm_model),
            'messages':[{'role':'system','content':'You must respond with a JSON object.'},
                        {'role':'user','content':prompt}]}
        key=hashlib.sha256(json.dumps({'request':payload,'schema':response_format},sort_keys=True).encode()).hexdigest()
        cache=self.path/'completions'/f'{key}.json'
        if cache.exists():return json.loads(cache.read_text(encoding='utf-8'))['content']
        try:
            schema=(response_format or {}).get('json_schema',{}).get('schema',{})
            role='amem_evolution' if 'should_evolve' in schema.get('properties',{}) else 'amem_metadata'
            response=self.client.post(self.client.settings.llm_url.rstrip('/')+'/chat/completions',
                                      self.client.settings.llm_key,payload,role)
            content=response['choices'][0]['message']['content']
            import jsonschema
            parsed=json.loads(content)
            if response_format and 'json_schema' in response_format:
                jsonschema.validate(parsed,response_format['json_schema']['schema'])
        except Exception as exc:
            raise AMemProviderStop('A-Mem provider/structured output failed: '+type(exc).__name__) from None
        cache.parent.mkdir(parents=True,exist_ok=True)
        cache.write_text(json.dumps({'content':content},ensure_ascii=False),encoding='utf-8')
        return content


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
