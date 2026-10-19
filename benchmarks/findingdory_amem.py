"""User hypergraph vs pinned official A-Mem system, 24 freshly generated answers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from threading import Event

from benchmarks.amem_adapter import create_memory,save_state,source_audit,observation_content,evidence as amem_evidence,ROOT,REVISION
from benchmarks.findingdory_memory import frame_times,validate_batch,structured_records,history_query,supported_frames
from benchmarks.findingdory_mem0 import ComparisonClients,code_hashes,read_rows
from benchmarks.findingdory_pilot import load,save,score_frames
from benchmarks.findingdory_structured import freeze,fingerprint
from benchmarks.findingdory_visual_answer import decode_visual,VISUAL_ANSWER_SYSTEM
from benchmarks.run_budget import RunBudget
from memory_agent.memory import HypergraphMemory
from memory_agent.recall import compact_record,parse_plan,pack_evidence
from memory_agent.settings import Settings


def report(run,tasks,budget):
    rows=read_rows(run)
    results={m:{'correct':sum(x['correct'] for x in rows if x['method']==m),
                'total':sum(x['method']==m for x in rows),
                'episodes':{ep:sum(x['correct'] for x in rows if x['method']==m and x['episode']==ep)
                            for ep in ('ep_1','ep_33')}} for m in ('hypergraph','amem')}
    out={'complete':all(v['total']==len(tasks) for v in results.values()),'results':results,
         'budget':budget.state,'new_tokens':budget.state['reported_tokens']-budget.state['carried_usage']['reported_tokens']}
    save(run/'summary.json',out)
    text=['# 超图记忆与A-Mem对比','',
          '相同192帧共同视觉观测、千问语言及嵌入模型、共同回答提示与图片筛选、Top-K=20、上下文40KB、每题最多12张图。',
          '双方12题全部重新作答，未复用历史答案。视觉观测复用同一份已校验输入；'+
          ('超图从观测重新建库。' if load(run/'evaluation_config.json')['hypergraph_index']=='rebuilt' else '超图复用未变化的持久索引。')+
          'A-Mem每帧一条笔记，分别维护两段历史。','',
          '| 方法 | 正确/已完成 | 第一段 | 第二段 |','|---|---:|---:|---:|']
    for m,v in results.items():text.append(f"| {m} | {v['correct']}/{v['total']} | {v['episodes']['ep_1']}/6 | {v['episodes']['ep_33']}/6 |")
    text.extend(['',f"新增reported tokens：{out['new_tokens']:,}；累计：{budget.state['reported_tokens']:,}；无累计token上限。",
        'A-Mem使用固定官方A-mem-sys源码，笔记分析、增强文本、关联与演化、search_agentic均保留。仅API传输和Chroma存储初始化适配当前端点、Qwen嵌入与运行目录；JSON schema由本地验证。',
        '这属于机器人视觉观测转文本后的官方系统场景适配，非论文原数据集或原嵌入配置复现。评分为12道开发题的离线目标帧命中，不能证明整体泛化优越性。',
        'API记录包含构建、嵌入、作答及最多一次约束复核。模型错误会停止，不以空元数据继续基线。'])
    (run/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,default=Path('runtime/findingdory_run_20261013_evidence_repair'))
    parser.add_argument('--run-dir',type=Path,default=Path('runtime/findingdory_run_20261013_amem'))
    parser.add_argument('--data-dir',type=Path,default=Path('runtime/findingdory'))
    parser.add_argument('--reuse-amem-state',action='store_true',help='Resume verified ingestion snapshots from source-run')
    parser.add_argument('--rebuild-hypergraph',action='store_true',help='Build a fresh hypergraph index from the shared observations')
    args=parser.parse_args();source,run,data=args.source_run,args.run_dir,args.data_dir
    if run.resolve() in {source.resolve(),data.resolve()}:raise ValueError('Isolated output required')
    if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=REVISION:
        raise ValueError('Official A-Mem revision changed')
    settings=Settings.load().model_copy(update={'timeout':120,'retries':0})
    old=load(source/'evaluation_config.json');models={k:getattr(settings,k) for k in old['models']}
    if models!=old['models']:raise ValueError('Models changed; rebuild input indices')
    tasks=load(data/'selected_tasks.json');shared={ep:load(source/'observations'/f'{ep}.json') for ep in ('ep_1','ep_33')}
    hashes=code_hashes()|{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        'benchmarks/amem_adapter.py','benchmarks/answer_normalization.py','benchmarks/findingdory_visual_answer.py','benchmarks/run_budget.py')}
    if fingerprint(tasks)!=old['task_sha256']:raise ValueError('Tasks changed')
    if not args.rebuild_hypergraph:
        for p in ('benchmarks/findingdory_memory.py','src/memory_agent/identity.py','src/memory_agent/models.py','src/memory_agent/semantic.py'):
            if hashes[p]!=old['code_sha256'][p]:raise ValueError('Hypergraph encoding changed')
    audit=source_audit()
    freeze(run/'evaluation_config.json',{'models':models,'source_run':str(source.resolve()),
        'task_sha256':fingerprint(tasks),'observation_sha256':{ep:fingerprint(raw) for ep,raw in shared.items()},
        'code_sha256':hashes,'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'amem_official':audit,'answer_prompt_sha256':fingerprint(VISUAL_ANSWER_SYSTEM),
        'methods':['hypergraph','amem'],'retrieval_k':20,'context_bytes':40000,'images_per_question':12,
        'ingestion_workers':2,'source_amem_state_resumed':args.reuse_amem_state,
        'hypergraph_index':'rebuilt' if args.rebuild_hypergraph else 'reused',
        'answers':'all 24 regenerated, common ANY-one normalization and bounded repair',
        'amem_note_unit':'one original observation per note; no query or reference answer',
        'amem_defaults':{'neighbor_k':5,'evolution_threshold':100,'temperature':1.0},
        'adaptations':['budgeted Qwen provider','same Qwen API embedding for Chroma default metric',
                       'persistent Chroma and JSON snapshots','JSON-object transport with local original-schema validation',
                       '4096 output-token bound when upstream specifies no limit']})
    previous=load(source/'api_budget.json');budget=RunBudget(run,max_requests=1000,max_reported_tokens=previous['limits']['max_reported_tokens'])
    if 'carried_usage' not in budget.state:
        budget.state['reported_tokens']=previous['reported_tokens']
        budget.state['carried_usage']={'source_run':str(source.resolve()),'reported_tokens':previous['reported_tokens']}
        budget.state['historical_usage']=previous['historical_usage'];budget.save()
    for ep in shared:
        transfers=[(source/'observations'/f'{ep}.json',run/'observations'/f'{ep}.json')]
        if not args.rebuild_hypergraph:
            transfers.append((source/'memory'/f'{ep}_hypergraph.sqlite3',run/'memory'/f'{ep}_hypergraph.sqlite3'))
        for original,dest in transfers:
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copy2(original,dest)
        if fingerprint(load(run/'observations'/f'{ep}.json'))!=fingerprint(shared[ep]):raise ValueError('Stored input changed')
        if args.reuse_amem_state and (source/'amem'/ep).exists() and not (run/'amem'/ep).exists():
            if old.get('amem_official')!=audit:raise ValueError('Official baseline implementation changed')
            shutil.copytree(source/'amem'/ep,run/'amem'/ep)
    records={ep:structured_records(ep,raw['observations']) for ep,raw in shared.items()}
    graphs={ep:HypergraphMemory(run/'memory'/f'{ep}_hypergraph.sqlite3') for ep in shared}
    if args.rebuild_hypergraph:
        for ep,graph in graphs.items():
            with graph.connection() as db:
                existing={row[0] for row in db.execute('SELECT id FROM records')}
            expected={record.id for record in records[ep]}
            if not existing:
                graph.add_many(records[ep])
            elif existing!=expected:
                raise ValueError('Partial hypergraph index; use a new run directory')
    namespace=(settings.embedding_url or settings.llm_url).rstrip('/')+'|'+settings.embedding_model
    for graph in graphs.values():
        with graph.connection() as db:
            providers={r[0] for r in db.execute('SELECT DISTINCT provider FROM embedding_cache')}
            if providers and providers!={namespace}:raise ValueError('Embedding endpoint changed')
            if not providers and not args.rebuild_hypergraph:raise ValueError('Embedding cache missing')
    client=ComparisonClients(settings,budget);memories={};provenance={}
    try:
        # Create official systems sequentially; provider/retriever initialization
        # hooks are global. The independent episode instances then evolve in order
        # in separate workers. Each has <=96 notes, below consolidation threshold.
        for ep in shared:
            memories[ep]=create_memory(run/'amem'/ep,client)
        stop=Event()
        def ingest(ep):
            directory=run/'amem'/ep;memory=memories[ep]
            raw=shared[ep]['observations'];times=frame_times(validate_batch(raw,[f['frame_index'] for f in raw]))
            provenance[ep]={f"{ep}_frame_{f['frame_index']}":[f['frame_index']] for f in raw}
            try:
              for frame in raw:
                if stop.is_set():return
                note_id=f"{ep}_frame_{frame['frame_index']}";content=observation_content(frame)
                if note_id in memory.memories and memory.read(note_id).content!=content:
                    raise ValueError('Resumed original note content changed')
                if note_id not in memory.memories:
                    content=observation_content(frame)
                    memory.add_note(content,id=note_id,time=times[frame['frame_index']][0].strftime('%Y%m%d%H%M'))
                    note=memory.read(note_id)
                    if not note.keywords or not note.tags or note.context=='General':raise ValueError('A-Mem metadata extraction silently degraded')
                    if note.content!=content:raise ValueError('Original note content changed')
                    save_state(memory,directory)
                if (frame['frame_index']+1)%8==0:
                    print(json.dumps({'amem_episode':ep,'notes':len(memory.memories),'evolved_notes':memory.evo_cnt,
                                      'tokens':budget.state['reported_tokens']}),flush=True)
            except BaseException:
                stop.set()
                raise
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(ingest,ep) for ep in shared]):future.result()
        done={(r['row_idx'],r['method']) for r in read_rows(run)}
        for task in tasks:
            ep=task['ep_id'];plan=parse_plan(task['question'])
            for method in ('hypergraph','amem'):
                if (task['row_idx'],method) in done:continue
                start=time.perf_counter();client.calls.clear()
                if method=='hypergraph':
                    query=history_query(task['question'],ep,records[ep]);query.recall_plan=plan
                    chosen,retrieval=graphs[ep].search(query,20,strategy='hypergraph',embedder=client.embed,
                        embedding_namespace=namespace,minimum_similarity=settings.semantic_min_similarity)
                    if retrieval['semantic_backend']!='embedding':raise ValueError('Embedding fallback')
                    evidence=[{'id':r.id,'content':compact_record(r),'frame_indices':supported_frames([r])} for r in chosen[:20]]
                    resolution=retrieval['plan_evidence']
                else:
                    retrieved=memories[ep].search_agentic(task['question'],k=20)
                    if not retrieved:raise ValueError('Official A-Mem search returned empty results')
                    evidence=amem_evidence(memories[ep],retrieved,provenance[ep]);resolution=None
                    retrieval={'results':retrieved,'official_method':'search_agentic','revision':REVISION}
                evidence,resolution=pack_evidence(evidence,resolution,require_all_targets=plan.all_targets or bool(plan.visit_order))
                answer=decode_visual(client,task['question'],evidence,shared[ep]['observations'],data,ep,resolution)
                gold=json.loads(task['answer']);available={n for e in evidence for n in e['frame_indices']}
                row={'row_idx':task['row_idx'],'episode':ep,'method':method,'question':task['question'],
                    'prediction':answer['goal_frames'],'gold':gold,'correct':score_frames(answer['goal_frames'],gold,
                        task['low_level_category']=='Unordered Revisitation'),'answer':answer,'evidence':evidence,
                    'retrieval':retrieval,'packed_resolution':resolution,'calls':client.calls.copy(),'reused':False,
                    'latency_ms':round((time.perf_counter()-start)*1000,2),
                    'candidate_gold_coverage':sum(bool(available&set(g)) for g in gold)/len(gold)}
                with (run/'predictions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                print(json.dumps({'task':task['row_idx'],'method':method,'correct':row['correct'],'prediction':row['prediction'],
                                  'tokens':budget.state['reported_tokens']}),flush=True)
    finally:
        client.close();summary=report(run,tasks,budget)
        print(json.dumps({'complete':summary['complete'],'results':summary['results'],'tokens':budget.state['reported_tokens']}),flush=True)


if __name__=='__main__':main()
