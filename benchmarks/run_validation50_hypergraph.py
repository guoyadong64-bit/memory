"""Evaluate the existing hypergraph on all 2,901 questions using sealed observations.

No video captioning or downloads occur here. --check-only never sends model requests.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import hmac
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from benchmarks.comparison_methods import HypergraphMethod
from benchmarks.comparison_worker import CachedClients, atomic_json, score_task
from benchmarks.findingdory_memory import SCHEMA_VERSION, validate_batch
from benchmarks.findingdory_pilot import load
from benchmarks.findingdory_structured import fingerprint, freeze
from benchmarks.findingdory_visual_answer import decode_visual
from benchmarks.prepare_validation50_observations import DEFAULT_DATA, checksum, selected_data
from benchmarks.run_budget import RunBudget
from benchmarks.run_comparison import require_live_settings
from memory_agent.settings import Settings

DEFAULT_SHARED = REPO / "runtime/experiments/findingdory_validation50_perception_v2"
DEFAULT_RUN = REPO / "runtime/experiments/hypergraph_validation50"


def verify_inputs(data: Path, shared: Path):
    manifest = selected_data(data)
    seal_path = shared / "observations_manifest.json"
    if not seal_path.is_file():
        raise RuntimeError("50 段视觉观测尚未封存；请先完成 resume_validation50_observations")
    sealed = load(seal_path)
    config = load(shared / "config.json")
    if (sealed.get("protocol") != "findingdory-validation50-shared-perception-v1"
            or sealed.get("episodes") != manifest["episodes"]
            or sealed.get("dataset_revision") != manifest["revision"]
            or sealed.get("configuration_fingerprint") != config.get("fingerprint")
            or config.get("dataset") != manifest
            or set(sealed.get("observation_sha256", {})) != set(manifest["episodes"])):
        raise RuntimeError("共享观测封存清单与验证集不一致")
    observations = {}
    for number, ep in enumerate(manifest["episodes"], 1):
        path = shared / "observations" / f"{ep}.json"
        if checksum(path) != sealed["observation_sha256"][ep]:
            raise RuntimeError(f"封存观测被改动：{ep}")
        saved = load(path)
        frame_path = shared / "frames" / ep / "frames_manifest.json"
        if (saved.get("episode") != ep or saved.get("schema_version") != SCHEMA_VERSION
                or saved.get("configuration_fingerprint") != config["fingerprint"]
                or checksum(frame_path) != saved.get("frame_manifest_sha256")):
            raise RuntimeError(f"观测来源、结构或抽帧清单不一致：{ep}")
        validate_batch(saved["observations"], list(range(96)))
        frame_manifest = load(frame_path)
        video = f"videos/val/{ep}.mp4"
        expected = {f"{i:03d}.jpg" for i in range(96)}
        if (frame_manifest.get("source_video") != video
                or frame_manifest.get("video_sha256") != manifest["videos"][video]["sha256"]
                or set(frame_manifest.get("frame_sha256", {})) != expected):
            raise RuntimeError(f"原图来源不一致：{ep}")
        for name, sha in frame_manifest["frame_sha256"].items():
            if checksum(frame_path.parent / name) != sha:
                raise RuntimeError(f"原图被改动：{ep}/{name}")
        observations[ep] = saved["observations"]
        print(f"[本地校验] {number}/50 {ep}：96 帧观测与原图通过", flush=True)
    tasks = [json.loads(line) for line in
             (data / "validation_first50_questions.jsonl").read_text(encoding="utf-8").splitlines()]
    for task in tasks:
        if task["video"] != f"videos/val/{task['ep_id']}.mp4" or task["ep_id"] not in observations:
            raise RuntimeError("题目的视频归属不一致")
        gold = json.loads(task["answer"])
        # The official subsample includes [-1] and empty groups when no selectable
        # target frame exists. Preserve these questions instead of dropping them.
        if (not isinstance(gold, list) or not gold or any(
                not isinstance(group, list) or any(
                    type(n) is not int or not -1 <= n < 96 for n in group) for group in gold)):
            raise RuntimeError(f"官方目标帧标注格式异常：row_idx={task['row_idx']}")
    return manifest, tasks, observations


def has_valid_targets(task):
    return all(any(0 <= n < 96 for n in group) for group in json.loads(task["answer"]))


def score_result(task, answer, evidence):
    predicted = answer.get("goal_frames", [])
    valid = isinstance(predicted, list) and all(type(n) is int and 0 <= n < 96 for n in predicted)
    correct, coverage = score_task(task, answer if valid else {"goal_frames": []}, evidence)
    return bool(correct and valid), coverage


def frozen_config(run, data, shared, settings, manifest, tasks):
    salt_path = run / "credential_salt.bin"
    if not salt_path.exists():
        salt_path.write_bytes(os.urandom(32))
    salt = salt_path.read_bytes()
    if len(salt) != 32:
        raise RuntimeError("凭据指纹盐文件无效")
    names = ("run_validation50_hypergraph.py", "comparison_methods.py", "comparison_worker.py",
             "findingdory_memory.py", "findingdory_refinement.py", "findingdory_sequence.py",
             "findingdory_visual_answer.py", "findingdory_mem0.py", "findingdory_pilot.py",
             "findingdory_structured.py", "answer_normalization.py", "run_budget.py")
    paths = [REPO / "benchmarks" / name for name in names]
    paths.extend(sorted((REPO / "src/memory_agent").glob("*.py")))
    return {"protocol": "findingdory-validation50-hypergraph-v1", "method": "hypergraph",
            "data_directory": str(data), "shared_directory": str(shared),
            "dataset": manifest, "tasks_sha256": fingerprint(tasks),
            "observations_manifest_sha256": checksum(shared / "observations_manifest.json"),
            "models": {name: getattr(settings, name) for name in (
                "llm_url", "llm_model", "answer_model", "llm_thinking", "embedding_url",
                "embedding_model", "semantic_min_similarity")},
            "credential_fingerprints": {name: hmac.new(salt, key.encode(), hashlib.sha256).hexdigest()
                for name, key in {"llm": settings.llm_key,
                                  "embedding": settings.embedding_key or settings.llm_key}.items()},
            "source_sha256": {str(path.relative_to(REPO)): checksum(path) for path in paths},
            "max_answer_images": 12, "jev_enabled": False,
            "scoring": "exact goal count; frame membership per target; unordered bipartite match",
            "known_development_episodes": ["ep_1", "ep_33"]}


@contextmanager
def run_lock(run):
    """Prevent two terminals billing the same run concurrently; OS releases on crash."""
    with (run / ".run.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("此实验目录已有进程运行；请勿在两个终端同时续跑") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def load_answers(run, tasks):
    expected = {task["row_idx"]: task for task in tasks}
    found = {}
    for path in (run / "answers").glob("*.json"):
        row = load(path)
        index = row.get("row_idx")
        if (type(index) is not int or index not in expected or index in found
                or path.stem != str(index) or row.get("task_sha256") != fingerprint(expected[index])):
            raise RuntimeError(f"逐题断点与冻结题目不一致：{path}")
        correct, coverage = score_result(expected[index], row["prediction"], row["evidence"])
        if row["correct"] != correct or row["candidate_frame_covered"] != coverage:
            raise RuntimeError(f"逐题评分被改动：{path}")
        found[index] = row
    return found


def summarize(run, tasks, answers, budget, shared):
    rows = [answers[t["row_idx"]] for t in tasks if t["row_idx"] in answers]

    def stats(items, total):
        done = len(items)
        return {"correct": sum(r["correct"] for r in items), "completed": done, "total": total,
                "accuracy": sum(r["correct"] for r in items) / done if done else None,
                "candidate_frame_coverage": sum(r["candidate_frame_covered"] for r in items) / done if done else None}

    result = stats(rows, len(tasks))
    valid_tasks = [t for t in tasks if has_valid_targets(t)]
    result["with_valid_goal_frames"] = stats(
        [answers[t["row_idx"]] for t in valid_tasks if t["row_idx"] in answers], len(valid_tasks))
    result["without_valid_goal_frames"] = len(tasks) - len(valid_tasks)
    for field in ("ep_id", "high_level_category", "low_level_category"):
        groups = defaultdict(list)
        for task in tasks:
            groups[task[field]].append(task)
        result["by_" + field] = {name: stats([answers[t["row_idx"]] for t in group
                                              if t["row_idx"] in answers], len(group))
                                      for name, group in groups.items()}
    result["complete"] = len(rows) == len(tasks)
    result["api_usage"] = budget.state.copy()
    result["shared_perception_usage"] = load(shared / "perception/api_budget.json")
    if (shared / "source_reuse.json").exists():
        source = load(shared / "source_reuse.json")
        result["reused_source_perception_usage"] = {
            "requests_started": source["source_requests_started"],
            "reported_tokens": source["source_reported_tokens"]}
    atomic_json(run / "summary.json", result)
    accuracy = f"{result['accuracy']:.2%}" if rows else "—"
    coverage = f"{result['candidate_frame_coverage']:.2%}" if rows else "—"
    report = (f"# 超图记忆：FindingDory 前 50 段验证视频\n\n"
              f"状态：{'全部完成' if result['complete'] else '未完成，统计仅覆盖已答题目'}。\n\n"
              f"已答 {len(rows)}/{len(tasks)}，答对 {result['correct']}，正确率 {accuracy}，"
              f"候选帧覆盖率 {coverage}。\n\n"
              "正确率使用当前离线目标帧评分器：目标数量须正确，各目标帧须属于官方允许帧集合；"
              "有序题检查顺序，无序题做一对一匹配。候选帧覆盖率统计检索证据是否包含各目标的"
              "至少一个允许帧，不能替代答题正确率。本实验未运行 Habitat 导航，不报告导航 SR/SPL。\n\n"
              f"其中 {result['without_valid_goal_frames']} 道官方标注含空目标组或 -1 哨兵，"
              "不能通过选择有效帧完成，保留在全部题目的分母中；不把 -1 当成有效预测。"
              f"另列有有效目标帧的 {len(valid_tasks)} 道题统计于 summary.json 的 "
              "with_valid_goal_frames，便于区分选帧无解与方法失败。\n\n"
              f"共享观测：{shared}。本次不重新生成观测；答题时每题最多查看 12 张已有原图。"
              "记忆构建使用观测，不接收题目答案；仅评分器读取官方答案。当前实验路径不调用 Jev。\n\n"
              f"本方法请求 {budget.state['requests_started']} 次，报告 token "
              f"{budget.state['reported_tokens']}（输入与输出合计）；共享视觉前处理用量另列于 summary.json，"
              "不计入本方法新用量。失败但无 usage 的请求可能未计入 token，费用以平台账单为准。\n\n"
              "ep_1、ep_33 已参与之前的方法开发；不能把全部 50 段宣称为未见测试集。"
              "其他视频的独立性还需结合后续调参过程判断。\n\n"
              "| 类别 | 已答/总题 | 答对 | 正确率 |\n|---|---:|---:|---:|\n")
    for name, group in result["by_high_level_category"].items():
        acc = f"{group['accuracy']:.2%}" if group["completed"] else "—"
        report += f"| {name} | {group['completed']}/{group['total']} | {group['correct']} | {acc} |\n"
    (run / "REPORT.md").write_text(report, encoding="utf-8")
    if result["complete"]:
        (run / "predictions.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows), encoding="utf-8")
    return result


def evaluate(run, shared, tasks, observations, settings, *, max_requests=1_000_000,
             method_factory=HypergraphMethod, client_factory=CachedClients, decoder=decode_visual):
    budget = RunBudget(run, max_requests=max_requests, max_reported_tokens=0)
    answers = load_answers(run, tasks)
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task["ep_id"]].append(task)
    client = client_factory(settings.model_copy(update={"mode": "live", "timeout": 60, "retries": 0}),
                            budget, run / "response_cache")
    try:
        for ep, episode_tasks in grouped.items():
            pending = [t for t in episode_tasks if t["row_idx"] not in answers]
            if not pending:
                print(f"[复用] {ep} 全部 {len(episode_tasks)} 题已有断点", flush=True)
                continue

            def progress(**event):
                print(f"[超图记忆] {ep} {event['completed']}/{event['total']} 条记录", flush=True)

            method = method_factory(run, shared, {ep: observations[ep]}, client, progress)
            try:
                method.build()
                for task in pending:
                    index = task["row_idx"]
                    print(f"[检索与答题] {ep} row={index}，总进度 {len(answers)}/{len(tasks)}", flush=True)
                    started = time.monotonic()
                    client.calls.clear()
                    evidence, resolution, retrieval = method.retrieve(ep, task["question"])
                    answer = decoder(client, task["question"], evidence, observations[ep], shared, ep, resolution)
                    correct, covered = score_result(task, answer, evidence)
                    row = {"method": "hypergraph", "row_idx": index, "task_sha256": fingerprint(task),
                           "episode": ep, "task_id": task["task_id"], "question": task["question"],
                           "high_level_category": task["high_level_category"],
                           "low_level_category": task["low_level_category"],
                           "prediction": answer, "gold_goal_frames": json.loads(task["answer"]),
                           "correct": correct, "candidate_frame_covered": covered, "evidence": evidence,
                           "resolution": resolution, "retrieval": retrieval, "calls": client.calls.copy(),
                           "latency_seconds": round(time.monotonic() - started, 3)}
                    atomic_json(run / "answers" / f"{index}.json", row)
                    answers[index] = row
                    hits = sum(r["correct"] for r in answers.values())
                    print(f"[hypergraph·answer] {ep} {len(answers)}/{len(tasks)} | 正确 {hits}"
                          f" | 请求 {budget.state['requests_started']} | token {budget.state['reported_tokens']}", flush=True)
            finally:
                method.close()
            summarize(run, tasks, answers, budget, shared)
    finally:
        client.close()
        result = summarize(run, tasks, answers, budget, shared)
    (run / "last_failure.json").unlink(missing_ok=True)
    print(f"[完成] {result['correct']}/{result['total']}，正确率 {result['accuracy']:.2%}；"
          f"报告：{run / 'REPORT.md'}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--shared-perception-dir", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--max-requests", type=int, default=1_000_000)
    parser.add_argument("--check-only", action="store_true", help="校验全部输入与配置，不调用模型")
    args = parser.parse_args(argv)
    data, shared, run = (p.resolve() for p in (args.data_dir, args.shared_perception_dir, args.run_dir))
    if args.max_requests < 1:
        parser.error("请求上限须大于零")
    for left, right in ((data, shared), (data, run), (shared, run)):
        if left == right or left in right.parents or right in left.parents:
            parser.error("数据、共享观测和本方法结果目录须相互独立")
    print("[预检] 校验本地 50 段视频、4800 帧与题目清单，不发送模型请求", flush=True)
    settings = Settings.load(REPO / ".env")
    require_live_settings(settings)
    manifest, tasks, observations = verify_inputs(data, shared)
    valid_count = sum(has_valid_targets(t) for t in tasks)
    print(f"[预检通过] 50 段视频、{len(tasks)} 题，其中 {valid_count} 题有有效目标帧；复用封存观测", flush=True)
    if args.check_only:
        return
    run.mkdir(parents=True, exist_ok=True)
    with run_lock(run):
        freeze(run / "config.json", frozen_config(run, data, shared, settings, manifest, tasks))
        print("[真实模型] 后续检索会调用嵌入模型，答题会调用视觉大模型；累计 token 不设硬上限", flush=True)
        try:
            evaluate(run, shared, tasks, observations, settings, max_requests=args.max_requests)
        except BaseException as exc:
            # Store the exception class only; provider details may contain credentials.
            atomic_json(run / "last_failure.json", {"type": type(exc).__name__})
            print(f"[中断] {type(exc).__name__}；已保存断点，同一命令可续跑：{run}", flush=True)
            raise


if __name__ == "__main__":
    main()
