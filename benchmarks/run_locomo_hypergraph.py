"""Run the hypergraph on pinned LoCoMo conversations; default mode calls real APIs.

--prepare-only downloads public data only. --check-only makes no model requests.
Existing FindingDory code, settings files, data and experiment outputs are untouched.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from benchmarks.comparison_worker import CachedClients
from benchmarks.locomo_data import (CATEGORIES, DEFAULT_DATA, FILES, REVISION, UPSTREAM,
                                    OfficialScorer, atomic_json, checksum, digest, load,
                                    prepare, select_tasks)
from benchmarks.locomo_hypergraph import LoCoMoHypergraph
from benchmarks.run_budget import RunBudget
from benchmarks.run_comparison import require_live_settings
from benchmarks.run_validation50_hypergraph import run_lock
from memory_agent.settings import Settings
from memory_agent.semantic import validate_vectors

DEFAULT_RUN = REPO / "runtime/experiments/hypergraph_locomo"


class LoCoMoClients(CachedClients):
    """Reuse durable HTTP cache/accounting; bound embedding provider batches."""
    def post(self, url, key, payload, role):
        key_hash = hashlib.sha256(json.dumps(
            {"url": url, "role": role, "payload": payload}, sort_keys=True,
            ensure_ascii=False).encode()).hexdigest()
        if not (self.cache_dir / f"{key_hash}.json").exists():
            print(f"[模型请求] {role} | 请求 {self.budget.state['requests_started'] + 1} | 等待响应", flush=True)
        result = super().post(url, key, payload, role)
        return result

    def embed(self, texts):
        # A query first embedded together with uncached memories must still be
        # reusable when resumption requests that query alone. HTTP batch caching
        # by itself cannot handle that change in batch composition.
        model_scope = [(self.settings.embedding_url or self.settings.llm_url).rstrip("/"),
                       self.settings.embedding_model]
        keys = {text: digest([*model_scope, text]) for text in texts}
        found, missing = {}, []
        for text, key in keys.items():
            path = self.cache_dir / "embedding_vectors" / (key + ".json")
            if path.exists():
                found[text] = validate_vectors([load(path)], 1)[0]
            else:
                missing.append(text)
        for start in range(0, len(missing), 20):
            batch = missing[start:start + 20]
            vectors = validate_vectors(super().embed(batch), len(batch))
            for text, vector in zip(batch, vectors):
                atomic_json(self.cache_dir / "embedding_vectors" / (keys[text] + ".json"), vector)
                found[text] = vector
        return validate_vectors([found[text] for text in texts], len(texts))


def configuration(data, views, tasks, settings, options):
    # Freeze implementation and input/model choices, never store credentials.
    paths = [Path(__file__), REPO / "benchmarks/locomo_data.py", REPO / "benchmarks/locomo_hypergraph.py",
             REPO / "benchmarks/comparison_worker.py", REPO / "benchmarks/run_budget.py",
             REPO / "benchmarks/run_validation50_hypergraph.py", REPO / "benchmarks/run_comparison.py"]
    paths.extend(sorted((REPO / "src/memory_agent").glob("*.py")))
    return {"protocol": "locomo-hypergraph-text-v1", "method": "hypergraph_locomo",
            "dataset": {"repository": UPSTREAM, "revision": REVISION,
                        "file_sha256": checksum(data / "locomo10.json"), "directory": str(data)},
            "scorer": {"source": "official/task_eval/evaluation.py", "sha256": FILES["official/evaluation.py"][1],
                       "metric": "official Porter-stemmed category-aware token F1; question-weighted categories 1-4",
                       "nltk_version": version("nltk"), "adversarial_metric": "official abstention rule, reported separately"},
            "conversation_sha256": digest(views), "tasks_sha256": digest(tasks),
            "conversations": list(views), "questions": len(tasks), "options": options,
            "models": {name: getattr(settings, name) for name in (
                "llm_url", "llm_model", "understanding_model", "answer_model", "llm_thinking",
                "embedding_url", "embedding_model", "semantic_min_similarity")},
            "implementation_sha256": {str(path.relative_to(REPO)): checksum(path) for path in paths},
            "memory": "existing SQLite HypergraphMemory plus dialogue/event adapter and bounded incidence traversal",
            "time_convention": "dataset-local session dates; UTC storage anchor and microsecond turn order are not measured times",
            "images": "provided BLIP captions only; image URLs and pixels are not requested",
            "annotations": "QA/answers/gold evidence/event_summary are excluded from memory, retrieval and answer model input",
            "jev_enabled": False, "reported_token_limit": None,
            "comparability": "this text/caption adapter has not been established as identical to any published baseline protocol"}


def freeze_config(run, config):
    frozen = config | {"fingerprint": digest(config)}
    path = run / "config.json"
    if path.exists():
        if load(path) != frozen:
            raise RuntimeError("模型、数据、代码或实验参数已变化；请使用新的 --run-dir，不能混用旧结果")
    elif any(run.glob("answers/*.json")) or any(run.glob("memory/*.sqlite3")):
        raise RuntimeError("输出目录已有结果但缺少配置；请使用新的 --run-dir")
    else:
        atomic_json(path, frozen)
    return frozen


def load_answers(run, tasks, config, scorer):
    expected = {task["task_id"]: task for task in tasks}
    rows = {}
    for path in sorted((run / "answers").glob("*.json")):
        row = load(path)
        task = expected.get(row.get("task_id"))
        if (task is None or path.stem != task["task_id"] or row.get("task_sha256") != digest(task)
                or row.get("configuration_fingerprint") != config["fingerprint"]):
            raise RuntimeError(f"答题断点不属于本次冻结配置：{path.name}")
        prediction = row.get("prediction")
        if not isinstance(prediction, dict) or not isinstance(prediction.get("answer"), str):
            raise RuntimeError(f"答题断点格式异常：{path.name}")
        row["score"] = scorer.score(prediction["answer"], task)
        row["evidence_recall"] = scorer.evidence_recall(task, row["retrieval"]["context_ids"])
        rows[task["task_id"]] = row
    return rows


def make_summary(rows, tasks, budget, active_seconds):
    completed = list(rows.values())
    answerable = [row for row in completed if row["category"] != 5]
    coverage = [row["evidence_recall"] for row in completed if row["evidence_recall"] is not None]
    categories = {}
    for category in sorted({task["category"] for task in tasks}):
        group = [row for row in completed if row["category"] == category]
        categories[str(category)] = {"name": CATEGORIES[category], "completed": len(group),
                                     "total": sum(task["category"] == category for task in tasks),
                                     "metric": "abstention_accuracy" if category == 5 else "token_f1",
                                     "score": sum(row["score"] for row in group) / len(group) if group else None}
    return {"method": "hypergraph_locomo", "complete": len(completed) == len(tasks),
            "completed": len(completed), "total": len(tasks),
            "overall_f1": sum(row["score"] for row in answerable) / len(answerable) if answerable else None,
            "overall_f1_questions": len(answerable), "overall_f1_excludes_category5": True,
            "by_category": categories,
            "gold_dialog_evidence_recall": sum(coverage) / len(coverage) if coverage else None,
            "evidence_recall_scored_questions": len(coverage),
            "api_usage": dict(budget.state), "active_seconds": round(active_seconds, 3),
            "completed_qa_seconds": round(sum(row["latency_seconds"] for row in completed), 3),
            "interpretation": "F1 is graded token overlap, not the fraction of fully correct answers. "
                              "Evidence recall uses original dialogue IDs, not FindingDory frame coverage. "
                              "Active time includes build/retrieval/answer waits and excludes downtime between invocations. "
                              "Provider failures can have unreported/billed usage; platform billing is authoritative."}


def write_results(run, tasks, rows, summary):
    atomic_json(run / "summary.json", summary)
    output = run / "predictions.jsonl"
    temp = run / "predictions.jsonl.tmp"
    try:
        with temp.open("w", encoding="utf-8") as stream:
            for task in tasks:
                if task["task_id"] in rows:
                    stream.write(json.dumps(rows[task["task_id"]], ensure_ascii=False) + "\n")
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    score = summary["overall_f1"]
    text = ["# LoCoMo 超图记忆实验", "", f"完成状态：{'完成' if summary['complete'] else '未完成，可续跑'}；"
            f"进度 {summary['completed']}/{summary['total']}。", "",
            f"四类可回答问题的总体 F1：{score:.4f}（{score * 100:.2f} 分）。" if score is not None else "总体 F1：尚无结果。",
            "", "| 类别 | 完成/题量 | 指标 | 得分（0—100） |", "|---|---:|---|---:|"]
    for group in summary["by_category"].values():
        value = f"{group['score'] * 100:.2f}" if group["score"] is not None else "—"
        text.append(f"| {group['name']} | {group['completed']}/{group['total']} | {group['metric']} | {value} |")
    recall = summary["gold_dialog_evidence_recall"]
    usage = summary["api_usage"]
    text += ["", f"原始对话证据召回率：{recall:.2%}。" if recall is not None else "原始对话证据召回率：无可计分标注。",
             f"累计 API 请求：{usage['requests_started']}；平台报告 token：{usage['reported_tokens']}。",
             f"累计活动时间：{summary['active_seconds'] / 3600:.3f} 小时；已完成题目的检索与答题时间："
             f"{summary['completed_qa_seconds'] / 3600:.3f} 小时。", "",
             "主指标使用固定版本官方评分函数的 Porter 词干化 token F1；多跳按官方逗号分项规则处理，"
             "开放域按官方分号规则处理。不可回答题若启用，单列官方拒答得分，不混入四类总体 F1。",
             "F1 不是完全答对比例；对话证据召回率不是候选帧覆盖率。只有实际传入回答器的原始对话 ID 计入证据召回。",
             "", "记忆仅使用 conversation 的原始文本、会话日期和官方提供的图片描述。"
             "未使用 QA、参考答案、标注证据、人工事件摘要或官方生成的会话摘要构建记忆。"
             "verified 标志代表本实验的入库与来源校验，不代表独立事实核验。",
             "本实验复用现有 SQLite 超图实现，新增对话事件适配和有界超边关联扩展；"
             "没有使用 FindingDory 专用帧选择器或 JEV。源对话的发言顺序不被当作现实事件发生顺序。",
             "", "比较其他论文时必须对齐模型、数据版本、问题范围、输入和评分；"
             "本报告不自动宣称与任何已发表数值处于同一协议。",
             "活动时间包含构建、模型等待与答题，不包含两次手动续跑之间的停机时间。"
             "token 为平台已返回的用量，失败请求可能没有用量回执，最终费用以平台账单为准。", "",
             f"数据及评分来源：[LoCoMo 官方仓库]({UPSTREAM}/tree/{REVISION})。", ""]
    (run / "REPORT.md").write_text("\n".join(text), encoding="utf-8")


def evaluate(run, views, tasks, settings, config, scorer, *, client_factory=LoCoMoClients,
             method_factory=LoCoMoHypergraph):
    run = Path(run)
    rows = load_answers(run, tasks, config, scorer)
    budget = RunBudget(run, max_requests=1_000_000, max_reported_tokens=0)
    client = client_factory(settings.model_copy(update={"mode": "live", "retries": 0,
                                                       "timeout": max(settings.timeout, 60)}),
                            budget, run / "response_cache")
    active_path = run / "activity.json"
    previous_active = load(active_path)["active_seconds"] if active_path.exists() else 0
    started = time.monotonic()

    def progress(**event):
        atomic_json(active_path, {"active_seconds": previous_active + time.monotonic() - started})
        name = {"memory_request": "记忆构建请求", "memory": "记忆构建"}[event["stage"]]
        print(f"[{name}] {event['conversation']} {event['completed']}/{event['total']} 批 | "
              f"请求 {budget.state['requests_started']} | token {budget.state['reported_tokens']}", flush=True)

    method = method_factory(run, client, progress=progress, **config["options"]["retrieval"])
    by_conversation = defaultdict(list)
    for task in tasks:
        by_conversation[task["sample_id"]].append(task)
    try:
        for index, (sample_id, selected) in enumerate(by_conversation.items(), 1):
            remaining = [task for task in selected if task["task_id"] not in rows]
            if not remaining:
                print(f"[复用完成断点] {index}/{len(views)} {sample_id}，{len(selected)} 题", flush=True)
                continue
            print(f"[会话] {index}/{len(views)} {sample_id}，原始发言 {len(views[sample_id]['turns'])}，待答 {len(remaining)} 题", flush=True)
            method.build(views[sample_id])
            for task in remaining:
                print(f"[检索与答题] {sample_id} {task['task_id']} | 总进度 {len(rows)}/{len(tasks)}", flush=True)
                question_started = time.monotonic()
                client.calls.clear()
                # Strict boundary: only a question string and conversation ID.
                evidence, retrieval = method.retrieve(sample_id, task["question"])
                prediction = method.answer(sample_id, task["question"], evidence, task["task_id"])
                row = {"method": "hypergraph_locomo", "task_id": task["task_id"],
                       "task_sha256": digest(task), "configuration_fingerprint": config["fingerprint"],
                       "sample_id": sample_id, "question": task["question"], "category": task["category"],
                       "reference_answer": task["answer"], "prediction": prediction,
                       "score": scorer.score(prediction["answer"], task),
                       "evidence_recall": scorer.evidence_recall(task, retrieval["context_ids"]),
                       "evidence": evidence, "retrieval": retrieval, "calls": client.calls.copy(),
                       "latency_seconds": round(time.monotonic() - question_started, 3)}
                atomic_json(run / "answers" / (task["task_id"] + ".json"), row)
                rows[task["task_id"]] = row
                elapsed = previous_active + time.monotonic() - started
                atomic_json(active_path, {"active_seconds": elapsed})
                summary = make_summary(rows, tasks, budget, elapsed)
                atomic_json(run / "summary.json", summary)
                value = summary["overall_f1"]
                display = f"{value * 100:.2f}" if value is not None else "—"
                print(f"[答题完成] {len(rows)}/{len(tasks)} | 当前 F1 {display} | "
                      f"请求 {budget.state['requests_started']} | token {budget.state['reported_tokens']}", flush=True)
    finally:
        elapsed = previous_active + time.monotonic() - started
        atomic_json(active_path, {"active_seconds": elapsed})
        try:
            client.close()
        finally:
            write_results(run, tasks, rows, make_summary(rows, tasks, budget, elapsed))
    (run / "last_failure.json").unlink(missing_ok=True)
    return load(run / "summary.json")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--categories", default="1,2,3,4", help="1多跳,2时间,3开放域,4单跳,5不可回答")
    parser.add_argument("--max-conversations", type=int, default=10)
    parser.add_argument("--max-questions", type=int, default=None, help="最多总题数；限制题量时请使用独立 --run-dir")
    parser.add_argument("--batch-size", type=int, default=20, help="每次记忆提取最多发言数；不跨会话")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--evidence-records", type=int, default=32)
    parser.add_argument("--evidence-chars", type=int, default=32000)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="只下载并校验官方公开数据，不调用模型")
    mode.add_argument("--check-only", action="store_true", help="仅本地预检，不下载、不调用模型")
    args = parser.parse_args(argv)
    try:
        categories = tuple(sorted(set(int(value) for value in args.categories.split(","))))
    except ValueError:
        parser.error("--categories 必须为逗号分隔的类别数字")
    if (not categories or not set(categories) <= set(CATEGORIES) or not 1 <= args.max_conversations <= 10
            or args.max_questions is not None and args.max_questions < 1
            or not 1 <= args.batch_size <= 20 or not 1 <= args.top_k <= 100
            or not 1 <= args.evidence_records <= 64 or not 2000 <= args.evidence_chars <= 100000):
        parser.error("类别、题量或检索参数超出允许范围")
    data, run = args.data_dir.resolve(), args.run_dir.resolve()
    if data == run or data in run.parents or run in data.parents:
        parser.error("数据目录和实验输出必须独立")
    print("[预检] LoCoMo 官方数据、评分源码与模型配置；此阶段不调用模型", flush=True)
    samples = prepare(data, download=not args.check_only)
    scorer = OfficialScorer(data)
    views, tasks = select_tasks(samples, categories, args.max_conversations, args.max_questions)
    print(f"[数据] {len(views)} 段对话，{len(tasks)} 题 | 类别 {','.join(map(str, categories))}", flush=True)
    if args.prepare_only:
        print(f"[准备完成] {data}；模型请求 0", flush=True)
        return
    settings = Settings.load(REPO / ".env")
    require_live_settings(settings)
    options = {"categories": categories, "max_conversations": args.max_conversations,
               "max_questions": args.max_questions,
               "retrieval": {"batch_size": args.batch_size, "top_k": args.top_k,
                             "evidence_records": args.evidence_records, "evidence_chars": args.evidence_chars}}
    # JSON roundtrip normalizes tuples for consistent resumed configuration checks.
    options = json.loads(json.dumps(options))
    config = configuration(data, views, tasks, settings, options)
    print(f"[模型] 提取 {settings.understanding_model or settings.llm_model} | "
          f"答题 {settings.answer_model or settings.llm_model} | 嵌入 {settings.embedding_model} | JEV 不启用", flush=True)
    if args.check_only:
        if (run / "config.json").exists():
            saved = load(run / "config.json")
            if saved != config | {"fingerprint": digest(config)}:
                raise RuntimeError("输出目录配置与本次参数不一致，请用新的 --run-dir")
            rows = load_answers(run, tasks, saved, scorer)
            print(f"[断点] 已完成 {len(rows)}/{len(tasks)} 题", flush=True)
        print("[预检通过] 模型请求 0；本地检查不代表 API 在线可用性", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    with run_lock(run):
        config = freeze_config(run, config)
        atomic_json(run / "selected_tasks.json", tasks)
        print("[真实模型实验] 开始或续跑；记录累计用量，不设累计 token 硬上限", flush=True)
        try:
            summary = evaluate(run, views, tasks, settings, config, scorer)
        except BaseException as exc:
            atomic_json(run / "last_failure.json", {"type": type(exc).__name__})
            print(f"[中断] {type(exc).__name__}；断点已保留，用同一命令续跑：{run}", flush=True)
            raise
    print(f"[完成] {summary['completed']}/{summary['total']} | 报告：{run / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
