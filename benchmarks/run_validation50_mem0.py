"""Standalone Mem0 evaluation on the same sealed validation-50 inputs as hypergraph.

Run with runtime/mem0-env/Scripts/python.exe; --check-only never calls model APIs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from benchmarks import run_validation50_hypergraph as common
from benchmarks.comparison_methods import Mem0Method
from benchmarks.comparison_worker import CachedClients, atomic_json
from benchmarks.findingdory_pilot import load
from benchmarks.findingdory_structured import fingerprint, freeze
from benchmarks.findingdory_visual_answer import decode_visual
from benchmarks.mem0_adapter import MEM0_VERSION
from benchmarks.run_budget import RunBudget
from benchmarks.run_comparison import require_live_settings
from memory_agent.settings import Settings

DEFAULT_RUN = REPO / "runtime/experiments/mem0_validation50"


def environment_check(run):
    if version("mem0ai") != MEM0_VERSION:
        raise RuntimeError(f"Mem0 SDK 应为 {MEM0_VERSION}，请使用 runtime/mem0-env/Scripts/python.exe")
    os.environ["MEM0_TELEMETRY"] = "false"
    # The official SDK creates its user/config files during import. Keep them in
    # this experiment instead of the user's home directory on Windows.
    os.environ["MEM0_DIR"] = str((run / "sdk_config").resolve())
    from mem0 import Memory
    from qdrant_client import QdrantClient
    return {name: version(name) for name in ("mem0ai", "qdrant-client", "pydantic", "httpx")}


def frozen_config(run, data, shared, settings, manifest, tasks, environment):
    config = common.frozen_config(run, data, shared, settings, manifest, tasks)
    config.update(protocol="findingdory-validation50-mem0-v1", method="mem0",
                  environment=environment, memory_algorithm="official add(infer=True) and search(top_k=20)")
    for path in (Path(__file__), REPO / "benchmarks/mem0_adapter.py"):
        config["source_sha256"][str(path.relative_to(REPO))] = common.checksum(path)
    return config


def verify_reference(shared, settings, tasks):
    """Fail before billing if a baseline would use different data/models/answer rules."""
    path = common.DEFAULT_RUN / "config.json"
    if not path.exists():
        return
    prior = load(path)
    current_models = {name: getattr(settings, name) for name in prior["models"]}
    if (prior["models"] != current_models or prior["tasks_sha256"] != fingerprint(tasks)
            or prior["observations_manifest_sha256"] != common.checksum(shared / "observations_manifest.json")):
        raise RuntimeError("当前模型、题目或共享观测与已完成的 hypergraph_validation50 不一致")
    frozen_sources = {name.replace("\\", "/"): sha for name, sha in prior["source_sha256"].items()}
    for name in ("benchmarks/findingdory_visual_answer.py", "benchmarks/answer_normalization.py",
                 "benchmarks/findingdory_pilot.py", "benchmarks/findingdory_mem0.py",
                 "benchmarks/run_validation50_hypergraph.py", "benchmarks/comparison_worker.py"):
        if common.checksum(REPO / name) != frozen_sources[name]:
            raise RuntimeError(f"共同答题、评分或输入校验代码已改变，不能混用旧超图结果：{name}")


def summarize(run, tasks, answers, budget, shared):
    # Reuse the exact same denominators, category aggregation and report body.
    result = common.summarize(run, tasks, answers, budget, shared)
    result.update(method="mem0", official_mem0_version=MEM0_VERSION)
    atomic_json(run / "summary.json", result)
    path = run / "REPORT.md"
    report = path.read_text(encoding="utf-8")
    title = "# 超图记忆：FindingDory 前 50 段验证视频"
    if not report.startswith(title):
        raise RuntimeError("共享报告模板已变化，请检查方法名称")
    path.write_text(report.replace(title, "# Mem0：FindingDory 前 50 段验证视频", 1) +
                    f"\nMem0 SDK {MEM0_VERSION}；沿用官方记忆提取、更新与检索，适配模型传输和帧来源。\n",
                    encoding="utf-8")
    return result


def evaluate(run, shared, tasks, observations, settings, *, max_requests=1_000_000,
             method_factory=Mem0Method, client_factory=CachedClients, decoder=decode_visual):
    budget = RunBudget(run, max_requests=max_requests, max_reported_tokens=0)
    answers = common.load_answers(run, tasks)
    if any(row.get("method") != "mem0" for row in answers.values()):
        raise RuntimeError("结果目录包含其他方法的断点；请使用独立的 --run-dir")
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task["ep_id"]].append(task)
    client = client_factory(settings.model_copy(update={"mode": "live", "timeout": 60, "retries": 0}),
                            budget, run / "response_cache")
    try:
        for ep, episode_tasks in grouped.items():
            pending = [t for t in episode_tasks if t["row_idx"] not in answers]
            if not pending:
                print(f"[复用] mem0 {ep} 全部 {len(episode_tasks)} 题已有断点", flush=True)
                continue

            def progress(**event):
                print(f"[mem0·memory] {ep} {event['completed']}/{event['total']} 帧"
                      f" | 请求 {budget.state['requests_started']} | token {budget.state['reported_tokens']}", flush=True)

            method = method_factory(run, shared, {ep: observations[ep]}, client, progress)
            try:
                print(f"[记忆] 开始 mem0 {ep}，复用视觉观测", flush=True)
                method.build()
                for task in pending:
                    index = task["row_idx"]
                    print(f"[检索与答题] mem0 {ep} row={index}，总进度 {len(answers)}/{len(tasks)}", flush=True)
                    started = time.monotonic()
                    client.calls.clear()
                    evidence, resolution, retrieval = method.retrieve(ep, task["question"])
                    answer = decoder(client, task["question"], evidence, observations[ep], shared, ep, resolution)
                    correct, covered = common.score_result(task, answer, evidence)
                    row = {"method": "mem0", "row_idx": index, "task_sha256": fingerprint(task),
                           "episode": ep, "task_id": task["task_id"], "question": task["question"],
                           "high_level_category": task["high_level_category"],
                           "low_level_category": task["low_level_category"],
                           "prediction": answer, "gold_goal_frames": json.loads(task["answer"]),
                           "correct": correct, "candidate_frame_covered": covered, "evidence": evidence,
                           "resolution": resolution, "retrieval": retrieval, "calls": client.calls.copy(),
                           "latency_seconds": round(time.monotonic() - started, 3)}
                    atomic_json(run / "answers" / f"{index}.json", row)
                    answers[index] = row
                    print(f"[mem0·answer] {ep} {len(answers)}/{len(tasks)}"
                          f" | 正确 {sum(r['correct'] for r in answers.values())}"
                          f" | 请求 {budget.state['requests_started']} | token {budget.state['reported_tokens']}", flush=True)
            finally:
                method.close()
            summarize(run, tasks, answers, budget, shared)
    finally:
        client.close()
        result = summarize(run, tasks, answers, budget, shared)
    (run / "last_failure.json").unlink(missing_ok=True)
    print(f"[完成] mem0 {result['correct']}/{result['total']}，正确率 {result['accuracy']:.2%}；"
          f"报告：{run / 'REPORT.md'}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=common.DEFAULT_DATA)
    parser.add_argument("--shared-perception-dir", type=Path, default=common.DEFAULT_SHARED)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--max-requests", type=int, default=1_000_000)
    parser.add_argument("--check-only", action="store_true", help="检查环境、模型配置和全部本地数据，不请求模型")
    args = parser.parse_args(argv)
    data, shared, run = (p.resolve() for p in (args.data_dir, args.shared_perception_dir, args.run_dir))
    if args.max_requests < 1:
        parser.error("请求上限须大于零")
    for left, right in ((data, shared), (data, run), (shared, run)):
        if left == right or left in right.parents or right in left.parents:
            parser.error("数据、共享观测和本方法结果目录须相互独立")
    print("[预检] Mem0 隔离环境与 50 段封存观测；此阶段不调用模型", flush=True)
    environment = environment_check(run)
    settings = Settings.load(REPO / ".env")
    require_live_settings(settings)
    manifest, tasks, observations = common.verify_inputs(data, shared)
    verify_reference(shared, settings, tasks)
    print(f"[预检通过] Mem0 {MEM0_VERSION}，50 段视频、{len(tasks)} 题；共同输入与超图实验一致", flush=True)
    if args.check_only:
        return
    run.mkdir(parents=True, exist_ok=True)
    with common.run_lock(run):
        freeze(run / "config.json", frozen_config(run, data, shared, settings, manifest, tasks, environment))
        print("[真实模型] Mem0 提取、嵌入、检索与统一答题会调用 API；复用原图与观测，不设累计 token 硬上限", flush=True)
        try:
            evaluate(run, shared, tasks, observations, settings, max_requests=args.max_requests)
        except BaseException as exc:
            atomic_json(run / "last_failure.json", {"type": type(exc).__name__})
            print(f"[中断] {type(exc).__name__}；断点保存在 {run}，同一命令可续跑", flush=True)
            raise


if __name__ == "__main__":
    main()
