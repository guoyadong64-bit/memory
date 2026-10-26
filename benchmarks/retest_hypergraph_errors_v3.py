"""Retest only baseline hypergraph errors using frozen observations and answer rules.

--check-only and --audit-only never call model APIs. A failed-only retest measures
repair on development errors; it is not a new overall/held-out accuracy estimate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from benchmarks import run_validation50_hypergraph as common
from benchmarks.comparison_worker import CachedClients, atomic_json
from benchmarks.findingdory_pilot import load
from benchmarks.findingdory_memory import structured_records
from benchmarks.findingdory_structured import fingerprint, freeze
from benchmarks.findingdory_visual_answer import decode_visual, select_images
from benchmarks.hypergraph_recall_v2 import project_retrieval as project_v2
from benchmarks.hypergraph_recall_v3 import HypergraphRecallV3, project_retrieval
from benchmarks.run_budget import RunBudget
from benchmarks.run_comparison import require_live_settings
from memory_agent.settings import Settings

DEFAULT_RUN = REPO / "runtime/experiments/hypergraph_validation50_errors_v3"


class RetestClients(CachedClients):
    """Read old embedding responses only; never reuse old answer responses."""
    def __init__(self, settings, budget, cache_dir, *, source_cache=None, transport=None):
        super().__init__(settings, budget, cache_dir, transport=transport)
        self.source_cache = Path(source_cache) if source_cache else None

    def post(self, url, key, payload, role):
        if role == "embedding" and self.source_cache:
            digest = hashlib.sha256(json.dumps({"url": url, "role": role, "payload": payload},
                sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            path = self.source_cache / f"{digest}.json"
            if path.is_file():
                self.calls.append({"role": role, "model": payload.get("model"), "cached": True,
                                   "source": "baseline_embedding_response"})
                return load(path)
        return super().post(url, key, payload, role)


def baseline_errors(source, tasks, shared, settings):
    config, summary = load(source / "config.json"), load(source / "summary.json")
    if (config.get("method") != "hypergraph" or not summary.get("complete")
            or config["tasks_sha256"] != fingerprint(tasks)
            or config["observations_manifest_sha256"] != common.checksum(shared / "observations_manifest.json")
            or config["models"] != {name: getattr(settings, name) for name in config["models"]}):
        raise RuntimeError("原实验未完成，或题目、共享观测、模型与本次不一致")
    frozen_sources = {name.replace("\\", "/"): sha for name, sha in config["source_sha256"].items()}
    for name in ("benchmarks/findingdory_visual_answer.py", "benchmarks/answer_normalization.py",
                 "benchmarks/findingdory_pilot.py", "benchmarks/findingdory_mem0.py",
                 "benchmarks/findingdory_memory.py", "benchmarks/run_validation50_hypergraph.py"):
        if common.checksum(REPO / name) != frozen_sources[name]:
            raise RuntimeError(f"共同观测、答题或评分实现已改变：{name}")
    answers = common.load_answers(source, tasks)
    if len(answers) != len(tasks) or any(row.get("method") != "hypergraph" for row in answers.values()):
        raise RuntimeError("原实验逐题断点不完整或包含其他方法")
    if sum(row["correct"] for row in answers.values()) != summary["correct"]:
        raise RuntimeError("原实验汇总与逐题评分不一致")
    selected = [task for task in tasks if not answers[task["row_idx"]]["correct"]]
    return selected, answers


def summarize(run, tasks, answers, baseline, budget):
    rows = [answers[t["row_idx"]] for t in tasks if t["row_idx"] in answers]
    completed, correct = len(rows), sum(row["correct"] for row in rows)
    indices = [row["row_idx"] for row in rows]
    result = {"method": "hypergraph_recall_v3", "scope": "previous_errors_only",
        "complete": completed == len(tasks), "completed": completed, "total": len(tasks),
        "correct": correct, "repair_rate": correct / completed if completed else None,
        "candidate_frame_coverage": sum(r["candidate_frame_covered"] for r in rows) / completed if completed else None,
        "baseline_coverage_on_same_completed_errors": sum(baseline[i]["candidate_frame_covered"] for i in indices) / completed if completed else None,
        "baseline_correct_not_retested": sum(row["correct"] for row in baseline.values()),
        "baseline_total": len(baseline), "new_overall_accuracy": None,
        "interpretation": "Only prior errors are retested. Baseline correct answers are not verified under the new method. "
                          "Do not report stitched answers as the new method's overall accuracy.",
        "candidate_limits": {"record_limit": 20, "evidence_bytes": 40000, "answer_images": 12,
                             "extra_entity_witness_frames": 8, "promoted_entities": 4},
        "api_usage": budget.state.copy()}
    groups = defaultdict(list)
    for row in rows:
        groups[row["low_level_category"]].append(row)
    result["by_category"] = {key: {"completed": len(group), "repaired": sum(r["correct"] for r in group),
        "repair_rate": sum(r["correct"] for r in group) / len(group)} for key, group in groups.items()}
    atomic_json(run / "summary.json", result)
    rate = f"{result['repair_rate']:.2%}" if completed else "—"
    coverage = f"{result['candidate_frame_coverage']:.2%}" if completed else "—"
    (run / "REPORT.md").write_text(
        "# 超图记忆优化：仅重测旧版错误题\n\n"
        f"已验证 {completed}/{len(tasks)}，修复 {correct} 题，错误子集修复率 {rate}，候选帧覆盖率 {coverage}。\n\n"
        "沿用封存视觉观测、原图、40KB证据预算、最多12张图及原答题评分规则。"
        "历史正确题没有重测；本结果不是新版完整正确率，也不是未见测试结果。"
        "包含官方无有效目标帧的旧错误题，不将其当作有效目标供模型参考。\n\n"
        f"本次新增请求 {budget.state['requests_started']}，报告 token {budget.state['reported_tokens']}。"
        "旧嵌入响应可复用；旧答题响应不导入；同一新实验目录的成功响应可在中断后复用。\n",
        encoding="utf-8")
    if result["complete"]:
        (run / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return result


def evaluate(run, source, shared, tasks, observations, settings, baseline, *, max_requests=1_000_000,
             method_factory=HypergraphRecallV3, client_factory=RetestClients, decoder=decode_visual):
    budget = RunBudget(run, max_requests=max_requests, max_reported_tokens=0)
    answers = common.load_answers(run, tasks)
    if any(row.get("method") != "hypergraph_recall_v3" for row in answers.values()):
        raise RuntimeError("新实验目录包含其他方法的答案")
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task["ep_id"]].append(task)
    client = client_factory(settings.model_copy(update={"mode": "live", "timeout": 60, "retries": 0}),
        budget, run / "response_cache", source_cache=source / "response_cache")
    try:
        for ep, group in grouped.items():
            pending = [t for t in group if t["row_idx"] not in answers]
            if not pending:
                print(f"[复用] {ep} 的旧错误题已全部验证", flush=True)
                continue
            method = method_factory(run, shared, {ep: observations[ep]}, client,
                lambda **event: print(f"[超图记忆v2] {ep} {event['completed']}/{event['total']}", flush=True))
            try:
                method.build()
                for task in pending:
                    index = task["row_idx"]
                    print(f"[重测错误题] {ep} row={index} {len(answers)}/{len(tasks)}", flush=True)
                    started = time.monotonic()
                    client.calls.clear()
                    evidence, resolution, retrieval = method.retrieve(ep, task["question"])
                    answer = decoder(client, task["question"], evidence, observations[ep], shared, ep, resolution)
                    correct, coverage = common.score_result(task, answer, evidence)
                    row = {"method": "hypergraph_recall_v3", "row_idx": index, "task_sha256": fingerprint(task),
                        "episode": ep, "task_id": task["task_id"], "question": task["question"],
                        "high_level_category": task["high_level_category"], "low_level_category": task["low_level_category"],
                        "prediction": answer, "gold_goal_frames": json.loads(task["answer"]), "correct": correct,
                        "candidate_frame_covered": coverage, "evidence": evidence, "resolution": resolution,
                        "retrieval": retrieval, "calls": client.calls.copy(),
                        "latency_seconds": round(time.monotonic() - started, 3)}
                    atomic_json(run / "answers" / f"{index}.json", row)
                    answers[index] = row
                    result = summarize(run, tasks, answers, baseline, budget)
                    print(f"[hypergraph-v3·answer] {ep} {len(answers)}/{len(tasks)} | 修复 {result['correct']}"
                          f" | 覆盖 {result['candidate_frame_coverage']:.2%} | 新请求 {budget.state['requests_started']}"
                          f" | token {budget.state['reported_tokens']}", flush=True)
            finally:
                method.close()
    finally:
        client.close()
        result = summarize(run, tasks, answers, baseline, budget)
    (run / "last_failure.json").unlink(missing_ok=True)
    print(f"[完成] 修复 {result['correct']}/{len(tasks)} 道旧错误题；{run / 'REPORT.md'}", flush=True)
    return result


def audit(run, tasks, observations, baseline):
    """Replay stored retrieval seeds; labels are read only by the scoring helper."""
    groups = defaultdict(list)
    for task in tasks:
        groups[task["ep_id"]].append(task)
    results = []
    for ep, group in groups.items():
        records = structured_records(ep, observations[ep])
        for task in group:
            previous = baseline[task["row_idx"]]
            evidence, resolution, details = project_retrieval(records,
                list(previous["retrieval"]["scores"])[:20], task["question"], previous["retrieval"].get("plan_evidence"))
            old_evidence, old_resolution, _ = project_v2(records,
                list(previous["retrieval"]["scores"])[:20], task["question"], previous["retrieval"].get("plan_evidence"))
            _, coverage = common.score_result(task, {"goal_frames": []}, evidence)
            _, previous_coverage = common.score_result(task, {"goal_frames": []}, old_evidence)
            images = select_images(evidence, observations[ep], resolution)
            old_images = select_images(old_evidence, observations[ep], old_resolution)
            _, image_coverage = common.score_result(task, {"goal_frames": []}, [{"frame_indices": images}])
            _, old_image_coverage = common.score_result(task, {"goal_frames": []}, [{"frame_indices": old_images}])
            results.append({"row_idx": task["row_idx"], "old_coverage": previous["candidate_frame_covered"],
                            "v2_coverage": previous_coverage, "new_coverage": coverage,
                            "image_coverage": image_coverage, "v2_image_coverage": old_image_coverage,
                            "packing": details})
        print(f"[离线审计] {ep} {len(results)}/{len(tasks)}；未调用模型", flush=True)
    result = {"scope": "previous_errors_only", "model_requests": 0, "questions": len(results),
              "old_coverage": sum(r["old_coverage"] for r in results) / len(results) if results else None,
              "new_coverage": sum(r["new_coverage"] for r in results) / len(results) if results else None,
              "v2_coverage": sum(r["v2_coverage"] for r in results) / len(results) if results else None,
              "image_coverage": sum(r["image_coverage"] for r in results) / len(results) if results else None,
              "v2_image_coverage": sum(r["v2_image_coverage"] for r in results) / len(results) if results else None,
              "improved_over_v2": sum(r["new_coverage"] > r["v2_coverage"] for r in results),
              "regressed_from_v2": sum(r["new_coverage"] < r["v2_coverage"] for r in results),
              "improved": sum(r["new_coverage"] > r["old_coverage"] for r in results),
              "regressed": sum(r["new_coverage"] < r["old_coverage"] for r in results), "rows": results,
              "interpretation": "Stored retrieval seeds replayed without model answers; no accuracy claim."}
    atomic_json(run / "offline_coverage_audit.json", result)
    print(f"[离线审计完成] 覆盖率 旧版 {result['old_coverage']:.2%} → V2 {result['v2_coverage']:.2%}"
          f" → V3 {result['new_coverage']:.2%}；原图覆盖 {result['v2_image_coverage']:.2%}"
          f" → {result['image_coverage']:.2%}；不代表答题准确率", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=common.DEFAULT_RUN)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--data-dir", type=Path, default=common.DEFAULT_DATA)
    parser.add_argument("--shared-perception-dir", type=Path, default=common.DEFAULT_SHARED)
    parser.add_argument("--max-requests", type=int, default=1_000_000)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    source, run, data, shared = (p.resolve() for p in (args.source_run, args.run_dir, args.data_dir, args.shared_perception_dir))
    paths = (source, run, data, shared)
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(paths) for b in paths[i + 1:]):
        parser.error("原实验、新实验、数据和共享观测目录必须相互独立")
    if args.max_requests < 1:
        parser.error("请求上限须为正整数")
    settings = Settings.load(REPO / ".env")
    require_live_settings(settings)
    print("[预检] 校验封存观测和旧版逐题结果；不请求模型", flush=True)
    manifest, all_tasks, observations = common.verify_inputs(data, shared)
    tasks, baseline = baseline_errors(source, all_tasks, shared, settings)
    print(f"[预检通过] 只重测 {len(tasks)} 道旧错误题，旧正确 {len(all_tasks) - len(tasks)} 题不重跑", flush=True)
    if args.check_only or not tasks:
        return
    run.mkdir(parents=True, exist_ok=True)
    with common.run_lock(run):
        config = common.frozen_config(run, data, shared, settings, manifest, tasks)
        config.update(protocol="hypergraph-errors-only-entity-incidence-v3", method="hypergraph_recall_v3",
            source_run=str(source), source_config_sha256=common.checksum(source / "config.json"),
            source_answers_sha256=fingerprint(baseline), selected_row_indices=[t["row_idx"] for t in tasks],
            scope="previous_errors_only; no new overall accuracy", reuse_source_roles=["embedding"])
        for name in ("retest_hypergraph_errors_v3.py", "hypergraph_evidence_subgraph.py", "hypergraph_recall_v3.py", "hypergraph_recall_v2.py"):
            config["source_sha256"]["benchmarks/" + name] = common.checksum(REPO / "benchmarks" / name)
        freeze(run / "config.json", config)
        if args.audit_only:
            audit(run, tasks, observations, baseline)
            return
        print("[真实模型] 仅重测旧错误题；复用视觉观测和可用旧嵌入；答题调用API，不设累计token硬上限", flush=True)
        try:
            evaluate(run, source, shared, tasks, observations, settings, baseline, max_requests=args.max_requests)
        except BaseException as exc:
            atomic_json(run / "last_failure.json", {"type": type(exc).__name__})
            print(f"[中断] {type(exc).__name__}；同一命令续跑，断点：{run}", flush=True)
            raise


if __name__ == "__main__":
    main()
