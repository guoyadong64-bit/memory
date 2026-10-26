"""Standalone official A-Mem evaluation on the sealed FindingDory validation-50 inputs.

Run with runtime/amem-env/Scripts/python.exe; --check-only never calls model APIs.
"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from benchmarks import run_validation50_hypergraph as common
from benchmarks.amem_adapter import ROOT, REVISION, JSON_RECOVERY_POLICY, source_audit
from benchmarks.comparison_methods import AMemMethod
from benchmarks.comparison_worker import CachedClients, atomic_json
from benchmarks.findingdory_structured import fingerprint, freeze
from benchmarks.findingdory_visual_answer import decode_visual
from benchmarks.run_budget import RunBudget
from benchmarks.run_comparison import require_live_settings
from benchmarks.run_validation50_mem0 import verify_reference
from memory_agent.settings import Settings

DEFAULT_RUN = REPO / "runtime/experiments/amem_validation50"
LEGACY_JSON_RECOVERY_SOURCES = {
    'benchmarks/run_validation50_amem.py': '84d5f9d74dc9fd9d51ae5b214ca7a22e0857f90d7895c05288756fc8606dfba9',
    'benchmarks/amem_adapter.py': '976585d2d2c905dd323e9d46edbe09b87ba5b3f406795cff2a77767787f4be88',
}


def environment_check(run):
    if not ROOT.is_dir():
        raise RuntimeError(f"A-Mem 官方源码不存在：{ROOT}")
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise RuntimeError(f"A-Mem 官方源码应固定为 {REVISION}")
    audit = source_audit()
    # HEAD alone does not detect uncommitted modifications to the algorithm.
    for name in audit["sha256"]:
        original = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{REVISION}:{name}"])
        actual = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        if actual != original.replace(b"\r\n", b"\n"):
            raise RuntimeError(f"A-Mem 官方文件被修改：{name}")
    packages = {name: version(name) for name in (
        "chromadb", "jsonschema", "rank_bm25", "nltk", "litellm",
        "sentence-transformers", "pydantic", "httpx")}
    if packages["chromadb"] != "1.5.9":
        raise RuntimeError("请使用 runtime/amem-env/Scripts/python.exe；ChromaDB 应为 1.5.9")
    print("[源码与版本通过] 正在导入 A-Mem、ChromaDB 和 Transformer 依赖；不加载本地嵌入模型", flush=True)
    # Import dependencies now, before any paid request. Do not instantiate a
    # memory system here: its real storage belongs to the locked experiment.
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import agentic_memory.memory_system
    import chromadb
    import jsonschema
    return {"official_source": audit, "packages": packages}


def frozen_config(run, data, shared, settings, manifest, tasks, environment):
    config = common.frozen_config(run, data, shared, settings, manifest, tasks)
    config.update(
        protocol="findingdory-validation50-amem-v1", method="amem", environment=environment,
        memory_algorithm="official note analysis, linking, evolution and search_agentic",
        note_unit="one original shared observation per note; isolated memory per video",
        official_parameters={"neighbor_k": 5, "evolution_threshold": 100, "temperature": 1.0},
        retrieval_top_k=20, evidence_max_bytes=40000, query_writeback=False,
        amem_json_recovery=JSON_RECOVERY_POLICY,
        timestamp="existing frame_times mapping; ordering is not a real-world date",
        provenance="only original note content grants access to its source frame",
        adaptation="cached API transport, shared embedding service, persistent Chroma and JSON checkpoints")
    for path in (Path(__file__), REPO / "benchmarks/amem_adapter.py",
                 REPO / "benchmarks/run_validation50_mem0.py", REPO / "requirements-amem.txt"):
        config["source_sha256"][str(path.relative_to(REPO))] = common.checksum(path)
    return config


def freeze_run_config(run, config):
    """Allow only the exact known JSON-recovery predecessor, with a permanent audit."""
    path = run / 'config.json'
    expected = config | {'fingerprint': fingerprint(config)}
    if not path.exists():
        return freeze(path, config)
    prior = json.loads(path.read_text(encoding='utf-8'))
    if prior == expected:
        return prior
    predecessor = copy.deepcopy(config)
    predecessor.pop('amem_json_recovery', None)
    seen = set()
    for name in predecessor['source_sha256']:
        normalized = name.replace('\\', '/')
        if normalized in LEGACY_JSON_RECOVERY_SOURCES:
            predecessor['source_sha256'][name] = LEGACY_JSON_RECOVERY_SOURCES[normalized]
            seen.add(normalized)
    known_prior = predecessor | {'fingerprint': fingerprint(predecessor)}
    if seen != set(LEGACY_JSON_RECOVERY_SOURCES) or prior != known_prior:
        raise RuntimeError('Run configuration changed beyond the known A-Mem JSON repair; use a new --run-dir')
    backup = run / 'compatibility/config_before_amem_json_recovery_v1.json'
    audit = run / 'compatibility/amem_json_recovery_v1.json'
    if backup.exists() and json.loads(backup.read_text(encoding='utf-8')) != prior:
        raise RuntimeError('A-Mem recovery configuration backup mismatch')
    atomic_json(backup, prior)
    change = {'from_fingerprint': prior['fingerprint'], 'to_fingerprint': expected['fingerprint'],
              'reason': 'Complete and validate truncated JSON before official memory evolution; retain checkpoints',
              'changed_sources': {name: {'old': old, 'new': next(
                  sha for p, sha in config['source_sha256'].items() if p.replace('\\', '/') == name)}
                  for name, old in LEGACY_JSON_RECOVERY_SOURCES.items()},
              'recovery_policy': JSON_RECOVERY_POLICY}
    if audit.exists() and json.loads(audit.read_text(encoding='utf-8')) != change:
        raise RuntimeError('A-Mem recovery audit mismatch')
    atomic_json(audit, change)
    atomic_json(path, expected)
    print('[兼容续跑] 已记录 A-Mem JSON 截断恢复修复；保留原配置、记忆、答案与全部用量', flush=True)
    return expected


def summarize(run, tasks, answers, budget, shared):
    result = common.summarize(run, tasks, answers, budget, shared)
    result.update(method="amem", official_amem_source=source_audit())
    audit = run / 'compatibility/amem_json_recovery_v1.json'
    if audit.exists():
        result['compatibility_patch'] = json.loads(audit.read_text(encoding='utf-8'))
    atomic_json(run / "summary.json", result)
    path = run / "REPORT.md"
    report = path.read_text(encoding="utf-8")
    title = "# 超图记忆：FindingDory 前 50 段验证视频"
    if not report.startswith(title):
        raise RuntimeError("共享报告模板已变化，请检查方法名称")
    path.write_text(report.replace(title, "# A-Mem：FindingDory 前 50 段验证视频", 1) +
                    "\nA-Mem 使用固定提交的官方笔记分析、关联、演化与 search_agentic。"
                    "每帧共享观测对应一条笔记，每段视频独立建库；题目和答案不写回记忆。"
                    "仅适配 API 传输、统一嵌入服务、Chroma 持久化与检查点。"
                    "演化文本不额外获得来源帧，仅原始笔记提供相应原图。"
                    "这是视频任务适配，不是原论文对话基准的完全复现。\n" +
                    ("\n运行期间应用了有界 JSON 恢复修复，原配置及修复记录保存在 compatibility/；"
                     "已有结果保留，新增恢复请求计入本次用量。\n" if audit.exists() else ''), encoding="utf-8")
    return result


def evaluate(run, shared, tasks, observations, settings, *, max_requests=1_000_000,
             method_factory=AMemMethod, client_factory=CachedClients, decoder=decode_visual):
    budget = RunBudget(run, max_requests=max_requests, max_reported_tokens=0)
    answers = common.load_answers(run, tasks)
    if any(row.get("method") != "amem" for row in answers.values()):
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
                print(f"[复用] amem {ep} 全部 {len(episode_tasks)} 题已有断点", flush=True)
                continue

            def progress(**event):
                print(f"[amem·memory] {ep} {event['completed']}/{event['total']} 帧"
                      f" | 请求 {budget.state['requests_started']} | token {budget.state['reported_tokens']}", flush=True)

            method = method_factory(run, shared, {ep: observations[ep]}, client, progress)
            try:
                print(f"[记忆] 开始 amem {ep}，每帧分析与关联演化会调用模型；复用视觉观测", flush=True)
                method.build()
                for task in pending:
                    index = task["row_idx"]
                    print(f"[检索与答题] amem {ep} row={index}，总进度 {len(answers)}/{len(tasks)}", flush=True)
                    started = time.monotonic()
                    client.calls.clear()
                    evidence, resolution, retrieval = method.retrieve(ep, task["question"])
                    answer = decoder(client, task["question"], evidence, observations[ep], shared, ep, resolution)
                    correct, covered = common.score_result(task, answer, evidence)
                    row = {"method": "amem", "row_idx": index, "task_sha256": fingerprint(task),
                           "episode": ep, "task_id": task["task_id"], "question": task["question"],
                           "high_level_category": task["high_level_category"],
                           "low_level_category": task["low_level_category"],
                           "prediction": answer, "gold_goal_frames": json.loads(task["answer"]),
                           "correct": correct, "candidate_frame_covered": covered, "evidence": evidence,
                           "resolution": resolution, "retrieval": retrieval, "calls": client.calls.copy(),
                           "latency_seconds": round(time.monotonic() - started, 3)}
                    atomic_json(run / "answers" / f"{index}.json", row)
                    answers[index] = row
                    print(f"[amem·answer] {ep} {len(answers)}/{len(tasks)}"
                          f" | 正确 {sum(r['correct'] for r in answers.values())}"
                          f" | 请求 {budget.state['requests_started']} | token {budget.state['reported_tokens']}", flush=True)
            finally:
                method.close()
            summarize(run, tasks, answers, budget, shared)
    finally:
        client.close()
        result = summarize(run, tasks, answers, budget, shared)
    (run / "last_failure.json").unlink(missing_ok=True)
    print(f"[完成] amem {result['correct']}/{result['total']}，正确率 {result['accuracy']:.2%}；"
          f"报告：{run / 'REPORT.md'}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=common.DEFAULT_DATA)
    parser.add_argument("--shared-perception-dir", type=Path, default=common.DEFAULT_SHARED)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--max-requests", type=int, default=1_000_000)
    parser.add_argument("--check-only", action="store_true", help="检查环境、配置、源码和本地输入，不调用模型")
    args = parser.parse_args(argv)
    data, shared, run = (p.resolve() for p in (args.data_dir, args.shared_perception_dir, args.run_dir))
    if args.max_requests < 1:
        parser.error("请求上限须大于零")
    for left, right in ((data, shared), (data, run), (shared, run)):
        if left == right or left in right.parents or right in left.parents:
            parser.error("数据、共享观测和本方法结果目录须相互独立")
    print("[预检] A-Mem 官方源码、隔离环境和 50 段封存观测；依赖首次加载可能较慢，不调用模型", flush=True)
    environment = environment_check(run)
    settings = Settings.load(REPO / ".env")
    require_live_settings(settings)
    manifest, tasks, observations = common.verify_inputs(data, shared)
    verify_reference(shared, settings, tasks)
    print(f"[预检通过] A-Mem {REVISION[:12]}，50 段视频、{len(tasks)} 题；共同输入与超图实验一致", flush=True)
    if args.check_only:
        return
    run.mkdir(parents=True, exist_ok=True)
    with common.run_lock(run):
        freeze_run_config(run, frozen_config(run, data, shared, settings, manifest, tasks, environment))
        print("[真实模型] A-Mem 记忆构建、嵌入、检索和统一视觉答题会调用 API；不设累计 token 硬上限", flush=True)
        try:
            evaluate(run, shared, tasks, observations, settings, max_requests=args.max_requests)
        except BaseException as exc:
            atomic_json(run / "last_failure.json", {"type": type(exc).__name__})
            print(f"[中断] {type(exc).__name__}；断点保存在 {run}，同一命令可续跑", flush=True)
            raise


if __name__ == "__main__":
    main()
