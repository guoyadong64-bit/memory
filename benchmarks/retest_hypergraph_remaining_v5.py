"""V5 retest of verified errors remaining after the initial run and V4 repair."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))

from benchmarks import run_validation50_hypergraph as common
from benchmarks import retest_hypergraph_errors_v4 as previous
from benchmarks.comparison_worker import atomic_json
from benchmarks.findingdory_structured import fingerprint, freeze
from benchmarks.hypergraph_recall_v5 import HypergraphRecallV5, LIMITS
from benchmarks.hypergraph_visual_answer_v5 import decode_visual
from benchmarks.run_budget import RunBudget
from benchmarks.run_comparison import require_live_settings
from memory_agent.settings import Settings

DEFAULT_V4 = REPO/'runtime/experiments/hypergraph_validation50_errors_v4'
DEFAULT_RUN = REPO/'runtime/experiments/hypergraph_validation50_remaining_v5'


def remaining_errors(source, v4, tasks, shared, settings):
    old_errors, baseline = previous.baseline_errors(source, tasks, shared, settings)
    config, summary = common.load(v4/'config.json'), common.load(v4/'summary.json')
    if (config.get('method') != 'hypergraph_recall_v4' or not summary.get('complete')
            or config['tasks_sha256'] != fingerprint(old_errors)
            or config['observations_manifest_sha256'] != common.checksum(shared/'observations_manifest.json')
            or config['models'] != {name:getattr(settings,name) for name in config['models']}
            or Path(config['source_run']).resolve() != source.resolve()
            or config['source_config_sha256'] != common.checksum(source/'config.json')
            or config['source_answers_sha256'] != fingerprint(baseline)):
        raise RuntimeError('V4 results are incomplete or differ from the original run, inputs or models')
    repaired = common.load_answers(v4, old_errors)
    if (len(repaired) != len(old_errors) or len(repaired) != summary['completed']
            or any(r.get('method') != 'hypergraph_recall_v4' for r in repaired.values())
            or sum(r['correct'] for r in repaired.values()) != summary['correct']):
        raise RuntimeError('V4 checkpoints do not match their completed summary')
    current = {index:repaired.get(index,row) for index,row in baseline.items()}
    selected = [task for task in tasks if not current[task['row_idx']]['correct']]
    return selected, current


def summarize(run, tasks, answers, current, budget):
    rows = [answers[t['row_idx']] for t in tasks if t['row_idx'] in answers]
    completed, correct = len(rows), sum(r['correct'] for r in rows)
    preserved = sum(r['correct'] for r in current.values())
    result = {'method':'hypergraph_recall_v5','scope':'remaining_errors_after_initial_and_v4',
        'complete':completed==len(tasks), 'completed':completed, 'total':len(tasks), 'correct':correct,
        'repair_rate':correct/completed if completed else None,
        'candidate_frame_coverage':sum(r['candidate_frame_covered'] for r in rows)/completed if completed else None,
        'historical_correct_not_retested':preserved, 'original_total':len(current),
        'new_overall_accuracy':None, 'limits':LIMITS, 'api_usage':budget.state.copy(),
        'interpretation':'Only previously incorrect tasks are retested. Preserved answers do not constitute a full V5 evaluation.'}
    atomic_json(run/'summary.json',result)
    (run/'REPORT.md').write_text('# V5：仅重测当前仍错误的题目\n\n'
        f'保留历史答对的 {preserved} 题，本次已验证 {completed}/{len(tasks)}，修复 {correct} 题。\n\n'
        '复用封存观测与兼容旧嵌入，使用 V5 召回和原图选择；历史答案不进入记忆或回答模型。'
        '本结果是开发错误子集的修复结果，不是 V5 单版本的全量正确率。\n\n'
        f"新增请求 {budget.state['requests_started']}；输入输出报告 token {budget.state['reported_tokens']}。\n",
        encoding='utf-8')
    if result['complete']:
        (run/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    return result


def evaluate(run, source, shared, tasks, observations, settings, current, *, max_requests=1_000_000,
             method_factory=HypergraphRecallV5, client_factory=previous.RetestClients, decoder=decode_visual):
    budget = RunBudget(run,max_requests=max_requests,max_reported_tokens=0)
    answers = common.load_answers(run,tasks)
    if any(r.get('method') != 'hypergraph_recall_v5' for r in answers.values()):
        raise RuntimeError('Output contains results from a different method')
    groups = defaultdict(list)
    for task in tasks:
        groups[task['ep_id']].append(task)
    client = client_factory(settings.model_copy(update={'mode':'live','timeout':60,'retries':0}),budget,
                           run/'response_cache',source_cache=source/'response_cache')
    try:
        for ep, group in groups.items():
            pending = [t for t in group if t['row_idx'] not in answers]
            if not pending:
                print(f'[复用] {ep} 的本次剩余错误题已全部完成',flush=True)
                continue
            method = method_factory(run,shared,{ep:observations[ep]},client,
                lambda **e:print(f"[V5·记忆] {ep} {e['completed']}/{e['total']} 条记忆",flush=True))
            try:
                method.build()
                for task in pending:
                    index=task['row_idx']
                    print(f'[V5·重测剩余错误题] {ep} row={index} {len(answers)}/{len(tasks)}',flush=True)
                    start=time.monotonic()
                    client.calls.clear()
                    evidence,resolution,retrieval=method.retrieve(ep,task['question'])
                    answer=decoder(client,task['question'],evidence,observations[ep],shared,ep,resolution)
                    correct,coverage=common.score_result(task,answer,evidence)
                    row={'method':'hypergraph_recall_v5','row_idx':index,'task_sha256':fingerprint(task),
                        'episode':ep,'task_id':task['task_id'],'question':task['question'],
                        'high_level_category':task['high_level_category'],'low_level_category':task['low_level_category'],
                        'prediction':answer,'gold_goal_frames':json.loads(task['answer']),
                        'correct':correct,'candidate_frame_covered':coverage,'evidence':evidence,
                        'resolution':resolution,'retrieval':retrieval,'calls':client.calls.copy(),
                        'latency_seconds':round(time.monotonic()-start,3)}
                    atomic_json(run/'answers'/f'{index}.json',row)
                    answers[index]=row
                    result=summarize(run,tasks,answers,current,budget)
                    print(f"[V5·answer] {len(answers)}/{len(tasks)} | 修复 {result['correct']}"
                          f" | 请求 {budget.state['requests_started']} | token {budget.state['reported_tokens']}",flush=True)
            finally:
                method.close()
    finally:
        client.close()
        result=summarize(run,tasks,answers,current,budget)
    (run/'last_failure.json').unlink(missing_ok=True)
    print(f"[完成] V5 修复 {result['correct']}/{len(tasks)}；报告：{run/'REPORT.md'}",flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,default=common.DEFAULT_RUN)
    parser.add_argument('--v4-run',type=Path,default=DEFAULT_V4)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--data-dir',type=Path,default=common.DEFAULT_DATA)
    parser.add_argument('--shared-perception-dir',type=Path,default=common.DEFAULT_SHARED)
    parser.add_argument('--max-requests',type=int,default=1_000_000)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args(argv)
    source,v4,run,data,shared=(p.resolve() for p in (
        args.source_run,args.v4_run,args.run_dir,args.data_dir,args.shared_perception_dir))
    paths=(source,v4,run,data,shared)
    if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(paths) for b in paths[i+1:]):
        parser.error('Inputs, historical runs and new output must be independent directories')
    if args.max_requests<1:
        parser.error('Request limit must be positive')
    settings=Settings.load(REPO/'.env')
    require_live_settings(settings)
    print('[预检] 校验首轮、V4 和50段共享观测；不调用模型',flush=True)
    manifest,all_tasks,observations=common.verify_inputs(data,shared)
    tasks,current=remaining_errors(source,v4,all_tasks,shared,settings)
    print(f'[预检通过] 历史正确 {len(all_tasks)-len(tasks)} 题保留；仅重测当前仍错误的 {len(tasks)} 题',flush=True)
    if args.check_only or not tasks:
        return
    run.mkdir(parents=True,exist_ok=True)
    with common.run_lock(run):
        config=common.frozen_config(run,data,shared,settings,manifest,tasks)
        config.update(protocol='hypergraph-v5-remaining-errors',method='hypergraph_recall_v5',limits=LIMITS,
            scope='remaining_errors_after_initial_and_v4; no full V5 accuracy',source_run=str(source),v4_run=str(v4),
            source_config_sha256=common.checksum(source/'config.json'),v4_config_sha256=common.checksum(v4/'config.json'),
            current_answers_sha256=fingerprint(current),selected_row_indices=[t['row_idx'] for t in tasks],
            historical_correct_not_retested=len(all_tasks)-len(tasks))
        for name in ('retest_hypergraph_remaining_v5.py','hypergraph_recall_v5.py','hypergraph_visual_answer_v5.py',
                     'hypergraph_evidence_subgraph.py','hypergraph_recall_v3.py','hypergraph_recall_v2.py',
                     'retest_hypergraph_errors_v4.py'):
            config['source_sha256']['benchmarks/'+name]=common.checksum(REPO/'benchmarks'/name)
        freeze(run/'config.json',config)
        print('[真实模型] 只重测剩余错误题，复用观测；已正确题不发模型请求，不设累计token硬上限',flush=True)
        try:
            evaluate(run,source,shared,tasks,observations,settings,current,max_requests=args.max_requests)
        except BaseException as exc:
            atomic_json(run/'last_failure.json',{'type':type(exc).__name__})
            print(f'[中断] {type(exc).__name__}；同一命令续跑：{run}',flush=True)
            raise


if __name__=='__main__':
    main()
