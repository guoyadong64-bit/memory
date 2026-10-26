"""Full validation-50 V5 evaluation; --audit-only performs no model requests."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))

from benchmarks import run_validation50_hypergraph as common
from benchmarks.comparison_worker import atomic_json
from benchmarks.findingdory_memory import structured_records
from benchmarks.findingdory_structured import fingerprint, freeze
from benchmarks.hypergraph_recall_v5 import HypergraphRecallV5, WitnessIndex, LIMITS, select_images_v5
from benchmarks.hypergraph_visual_answer_v5 import decode_visual
from benchmarks.retest_hypergraph_errors_v4 import RetestClients
from benchmarks.run_comparison import require_live_settings
from benchmarks.run_validation50_mem0 import verify_reference
from memory_agent.settings import Settings

DEFAULT_RUN = REPO/'runtime/experiments/hypergraph_validation50_v5'


def source_rows(source, tasks, shared):
    config = common.load(source/'config.json')
    if (config['tasks_sha256'] != fingerprint(tasks) or
            config['observations_manifest_sha256'] != common.checksum(shared/'observations_manifest.json')):
        raise RuntimeError('Source retrieval seeds do not match sealed tasks and observations')
    rows = common.load_answers(source, tasks)
    if len(rows) != len(tasks):
        raise RuntimeError('Full baseline answer checkpoints are required for the offline audit')
    return rows


def audit(run, tasks, observations, baseline):
    rows = []
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task['ep_id']].append(task)
    for ep, group in grouped.items():
        index = WitnessIndex(structured_records(ep, observations[ep]))
        for task in group:
            old = baseline[task['row_idx']]
            evidence, resolution, details = index.project(list(old['retrieval']['scores'])[:20],
                task['question'], old['retrieval'].get('plan_evidence'))
            frames = {n for item in evidence for n in item['frame_indices']}
            images = select_images_v5(evidence, observations[ep], resolution)
            _, coverage = common.score_result(task, {'goal_frames': []}, evidence)
            _, image_coverage = common.score_result(task, {'goal_frames': []}, [{'frame_indices': images}])
            gold = json.loads(task['answer'])
            ceiling = sum(any(0 <= n < 96 for n in target) for target in gold)/len(gold)
            rows.append({'row_idx': task['row_idx'], 'episode': ep,
                'category': task['low_level_category'], 'valid_targets': common.has_valid_targets(task),
                'baseline_coverage': old['candidate_frame_covered'], 'candidate_coverage': coverage,
                'image_coverage': image_coverage, 'ceiling': ceiling, 'candidate_frames': sorted(frames),
                'selected_images': images, 'evidence_bytes': details['bytes'], 'evidence_records': len(evidence)})
        current = sum(r['candidate_coverage'] for r in rows)/len(rows)
        print(f'[V5 离线审计] {ep} {len(rows)}/{len(tasks)} | 候选覆盖 {current:.2%} | 模型请求 0', flush=True)
    def stats(items):
        return {'questions': len(items), 'candidate_frame_coverage': sum(r['candidate_coverage'] for r in items)/len(items),
                'image_coverage': sum(r['image_coverage'] for r in items)/len(items),
                'candidate_coverage_ceiling': sum(r['ceiling'] for r in items)/len(items)} if items else {}
    valid = [r for r in rows if r['valid_targets']]
    result = {'protocol': 'hypergraph-v5-full-development-audit', 'model_requests': 0, 'limits': LIMITS,
              'all_questions': stats(rows), 'with_valid_goal_frames': stats(valid),
              'baseline_candidate_coverage': sum(r['baseline_coverage'] for r in rows)/len(rows),
              'target': .95, 'target_met_all_questions': stats(rows)['candidate_frame_coverage'] >= .95,
              'interpretation': 'Full development-set seed replay; no new model answers and no accuracy claim.',
              'by_category': {name: stats([r for r in rows if r['category']==name]) for name in sorted({r['category'] for r in rows})},
              'rows': rows}
    atomic_json(run/'offline_coverage_audit.json', result)
    (run/'AUDIT.md').write_text(
        '# V5 全量开发题召回审计\n\n'
        f"全部 {len(rows)} 题：候选覆盖率 {result['all_questions']['candidate_frame_coverage']:.2%}，"
        f"实际12图覆盖率 {result['all_questions']['image_coverage']:.2%}；"
        f"全部可用帧的理论覆盖上限 {result['all_questions']['candidate_coverage_ceiling']:.2%}。\n\n"
        f"有有效目标帧的 {len(valid)} 题：候选覆盖率 {result['with_valid_goal_frames']['candidate_frame_coverage']:.2%}。\n\n"
        '0次模型请求；使用旧检索种子，按新规则重新组织证据。标签只由评分器读取。'
        '每题最多32个候选帧、20条视图、40KB证据及12张答题原图。'
        '这是已参与开发的数据，不能作为未见测试；覆盖率不是正确率。\n', encoding='utf-8')
    print(f"[审计完成] 全部题候选覆盖 {result['all_questions']['candidate_frame_coverage']:.2%}，"
          f"有解题覆盖 {result['with_valid_goal_frames']['candidate_frame_coverage']:.2%}；未调用模型", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=common.DEFAULT_DATA)
    parser.add_argument('--shared-perception-dir', type=Path, default=common.DEFAULT_SHARED)
    parser.add_argument('--source-run', type=Path, default=common.DEFAULT_RUN)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--max-requests', type=int, default=1_000_000)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    data, shared, source, run = (p.resolve() for p in (
        args.data_dir, args.shared_perception_dir, args.source_run, args.run_dir))
    paths = (data, shared, source, run)
    if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(paths) for b in paths[i+1:]):
        parser.error('Data, shared observations, source and V5 output must be independent')
    if args.max_requests < 1:
        parser.error('Request limit must be positive')
    print('[预检] 全量50段封存观测与原图；不发模型请求', flush=True)
    manifest, tasks, observations = common.verify_inputs(data, shared)
    baseline = source_rows(source, tasks, shared)
    if args.check_only:
        return
    run.mkdir(parents=True, exist_ok=True)
    source_files = ('hypergraph_recall_v5.py', 'hypergraph_visual_answer_v5.py', 'run_validation50_hypergraph_v5.py',
                    'hypergraph_evidence_subgraph.py', 'hypergraph_recall_v3.py', 'hypergraph_recall_v2.py',
                    'retest_hypergraph_errors_v4.py', 'findingdory_memory.py')
    with common.run_lock(run):
        if args.audit_only:
            freeze(run/'audit_config.json', {'protocol': 'hypergraph-v5-full-development-audit', 'limits': LIMITS,
                'tasks_sha256': fingerprint(tasks), 'shared_sha256': common.checksum(shared/'observations_manifest.json'),
                'source_config_sha256': common.checksum(source/'config.json'),
                'source_sha256': {name: common.checksum(REPO/'benchmarks'/name) for name in source_files}})
            audit(run, tasks, observations, baseline)
            return
        settings = Settings.load(REPO/'.env')
        require_live_settings(settings)
        verify_reference(shared, settings, tasks)
        config = common.frozen_config(run, data, shared, settings, manifest, tasks)
        config.update(protocol='hypergraph-v5-full-evaluation', method='hypergraph_v5', limits=LIMITS,
                      input_projection='query-conditioned incidence and independent original witness views',
                      image_selection='query-conditioned priority; original answer prompt and validation unchanged',
                      source_run=str(source), scope='all questions; no historical answer reuse')
        for name in source_files:
            config['source_sha256']['benchmarks/'+name] = common.checksum(REPO/'benchmarks'/name)
        freeze(run/'config.json', config)
        for row in common.load_answers(run,tasks).values():
            if 'read_projection_v5' not in row.get('retrieval',{}):
                raise RuntimeError('V5 output contains a historical/non-V5 answer; use an independent --run-dir')
        def clients(s, budget, cache):
            return RetestClients(s, budget, cache, source_cache=source/'response_cache')
        print('[真实模型] 全部2901题使用V5重新作答；复用观测和兼容旧嵌入，不导入旧答案', flush=True)
        try:
            common.evaluate(run, shared, tasks, observations, settings, max_requests=args.max_requests,
                            method_factory=HypergraphRecallV5, client_factory=clients, decoder=decode_visual)
        except BaseException as exc:
            atomic_json(run/'last_failure.json', {'type': type(exc).__name__})
            print(f'[中断] {type(exc).__name__}；同一命令续跑：{run}', flush=True)
            raise
        finally:
            if (run/'summary.json').exists():
                summary = common.load(run/'summary.json')
                summary.update(method='hypergraph_v5', scope='single-version full evaluation', limits=LIMITS)
                atomic_json(run/'summary.json', summary)


if __name__ == '__main__':
    main()
