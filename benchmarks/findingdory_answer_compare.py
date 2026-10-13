"""Regenerate both methods' answers from unchanged, verified memory indices.

The common visual decoder/selector may change; no historical answers are reused.
Only unchanged input memories and embeddings are copied to isolated storage.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from benchmarks.findingdory_memory import history_query,structured_records,supported_frames
from benchmarks.findingdory_mem0 import ComparisonClients,code_hashes,read_rows
from benchmarks.findingdory_pilot import load,save,score_frames
from benchmarks.findingdory_retest import report
from benchmarks.findingdory_structured import fingerprint,freeze
from benchmarks.findingdory_visual_answer import VISUAL_ANSWER_SYSTEM,decode_visual
from benchmarks.mem0_adapter import create_memory,memory_frame_indices
from benchmarks.run_budget import RunBudget
from memory_agent.memory import HypergraphMemory
from memory_agent.recall import parse_plan,compact_record,pack_evidence
from memory_agent.settings import Settings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,default=Path('runtime/findingdory_run_20261013_full_compare_verified'))
    parser.add_argument('--run-dir',type=Path,default=Path('runtime/findingdory_run_20261013_evidence_repair'))
    parser.add_argument('--data-dir',type=Path,default=Path('runtime/findingdory'))
    args=parser.parse_args();source,run,data=args.source_run,args.run_dir,args.data_dir
    if run.resolve() in {source.resolve(),data.resolve()}:raise ValueError('Use isolated output')
    settings=Settings.load().model_copy(update={'timeout':120,'retries':0})
    old=load(source/'evaluation_config.json')
    models={k:getattr(settings,k) for k in old['models']}
    if models!=old['models']:raise ValueError('Memory models changed; rebuild both indices')
    hashes=code_hashes() | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        'benchmarks/findingdory_visual_answer.py','benchmarks/answer_normalization.py','benchmarks/run_budget.py')}
    for p in ('benchmarks/findingdory_memory.py','benchmarks/mem0_adapter.py','src/memory_agent/identity.py',
              'src/memory_agent/models.py','src/memory_agent/semantic.py'):
        if hashes[p]!=old['code_sha256'][p]:raise ValueError('Memory encoding changed; rebuild '+p)
    tasks=load(data/'selected_tasks.json')
    if fingerprint(tasks)!=old['task_sha256']:raise ValueError('Tasks changed')
    shared={ep:load(source/'observations'/f'{ep}.json') for ep in ('ep_1','ep_33')}
    obs_hashes={ep:fingerprint(raw) for ep,raw in shared.items()}
    if obs_hashes!=old['observation_sha256']:raise ValueError('Shared observations changed')
    freeze(run/'evaluation_config.json',{'models':models,'source_run':str(source.resolve()),
        'observation_sha256':obs_hashes,'task_sha256':fingerprint(tasks),'code_sha256':hashes,
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'answer_prompt_sha256':fingerprint(VISUAL_ANSWER_SYSTEM),'retrieval_k':20,
        'image_limit':12,'max_context_utf8_bytes':40000,'max_answer_calls_per_question':2,
        'protocol':'unchanged verified indices copied, all 24 answers regenerated; common decoder and image selector',
        'historical_answers_reused':0})
    prior=load(source/'api_budget.json')
    budget=RunBudget(run,max_requests=120,max_reported_tokens=prior['limits']['max_reported_tokens'])
    if 'carried_usage' not in budget.state:
        budget.state['reported_tokens']=prior['reported_tokens']
        budget.state['carried_usage']={'source_run':str(source.resolve()),'reported_tokens':prior['reported_tokens']}
        budget.state['historical_usage']=prior['historical_usage'];budget.save()
    for ep in shared:
        for original,dest in ((source/'observations'/f'{ep}.json',run/'observations'/f'{ep}.json'),
                              (source/'memory'/f'{ep}_hypergraph.sqlite3',run/'memory'/f'{ep}_hypergraph.sqlite3')):
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copy2(original,dest)
        dest=run/'mem0'/ep
        if not dest.exists():shutil.copytree(source/'mem0'/ep,dest)
    save(run/'index_provenance.json',{'indices_copied_from':str(source.resolve()),
        'source_sdk_audit':load(source/'sdk_audit.json'),'observation_sha256':obs_hashes})
    records={ep:structured_records(ep,raw['observations']) for ep,raw in shared.items()}
    graphs={ep:HypergraphMemory(run/'memory'/f'{ep}_hypergraph.sqlite3') for ep in shared}
    namespace=(settings.embedding_url or settings.llm_url).rstrip('/')+'|'+settings.embedding_model
    for graph in graphs.values():
        with graph.connection() as db:
            if {x[0] for x in db.execute('SELECT DISTINCT provider FROM embedding_cache')}!={namespace}:
                raise ValueError('Embedding endpoint/model changed; rebuild indices')
    client=ComparisonClients(settings,budget);memories={}
    done={(r['row_idx'],r['method']) for r in read_rows(run)}
    try:
        dims=load(source/'embedding_dimensions.json')['dimensions']
        memories={ep:create_memory(run/'mem0'/ep,client,dims) for ep in shared}
        for task in tasks:
            ep=task['ep_id'];plan=parse_plan(task['question'])
            for method in ('hypergraph','mem0'):
                if (task['row_idx'],method) in done:continue
                start=time.perf_counter();client.calls.clear()
                if method=='hypergraph':
                    query=history_query(task['question'],ep,records[ep]);query.recall_plan=plan
                    selected,retrieval=graphs[ep].search(query,20,strategy='hypergraph',embedder=client.embed,
                        embedding_namespace=namespace,minimum_similarity=settings.semantic_min_similarity)
                    if retrieval['semantic_backend']!='embedding':raise ValueError('Embedding fallback')
                    evidence=[{'id':r.id,'content':compact_record(r),'frame_indices':supported_frames([r])} for r in selected[:20]]
                    resolution=retrieval['plan_evidence']
                else:
                    retrieval=memories[ep].search(task['question'],top_k=20,filters={'user_id':ep},explain=True)
                    evidence=[{'id':m['id'],'content':m['memory'],'frame_indices':memory_frame_indices(m)} for m in retrieval['results']]
                    resolution=None
                evidence,resolution=pack_evidence(evidence,resolution,require_all_targets=plan.all_targets or bool(plan.visit_order))
                answer=decode_visual(client,task['question'],evidence,shared[ep]['observations'],data,ep,resolution)
                gold=json.loads(task['answer']);available={f for item in evidence for f in item['frame_indices']}
                row={'row_idx':task['row_idx'],'episode':ep,'method':method,'question':task['question'],
                    'prediction':answer['goal_frames'],'gold':gold,
                    'correct':score_frames(answer['goal_frames'],gold,task['low_level_category']=='Unordered Revisitation'),
                    'answer':answer,'evidence':evidence,'retrieval':retrieval,'packed_resolution':resolution,
                    'reused':False,'calls':client.calls.copy(),'latency_ms':round((time.perf_counter()-start)*1000,2),
                    'candidate_gold_coverage':sum(bool(available & set(g)) for g in gold)/len(gold)}
                with (run/'predictions.jsonl').open('a',encoding='utf-8') as file:file.write(json.dumps(row,ensure_ascii=False)+'\n')
                print(json.dumps({'task':task['row_idx'],'method':method,'correct':row['correct'],
                    'prediction':row['prediction'],'tokens':budget.state['reported_tokens']}),flush=True)
    finally:
        for memory in memories.values():memory.close()
        client.close();summary=report(data,run)
        print(json.dumps({'complete':summary['complete'],'results':summary['results'],'tokens':budget.state['reported_tokens']}),flush=True)


if __name__=='__main__':main()
