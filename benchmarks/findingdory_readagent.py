"""Compare current hypergraph memory with pinned official ReadAgent on two clips.

ReadAgent's official pagination/gisting functions execute from its notebook.
Lookup uses its official prompt/page parsing with open-answer task adaptation.
Both methods share the existing visual goal-frame answerer and sampled images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from benchmarks.findingdory_memory import history_query, structured_records, supported_frames
from benchmarks.findingdory_mem0 import ComparisonClients, code_hashes, read_rows
from benchmarks.findingdory_pilot import load, save, score_frames
from benchmarks.findingdory_structured import fingerprint, freeze
from benchmarks.findingdory_visual_answer import VISUAL_ANSWER_SYSTEM, decode_visual
from benchmarks.readagent_official_adapter import OfficialReadAgent, official_source
from benchmarks.run_budget import RunBudget
from memory_agent.memory import HypergraphMemory
from memory_agent.recall import compact_record, pack_evidence, parse_plan
from memory_agent.settings import Settings


METHODS = ("hypergraph", "readagent")
EPISODES = ("ep_1", "ep_33")


def readagent_evidence(episode, pages, page_ids):
    return [{"id": f"{episode}_readagent_page_{page_id}",
             "content": {"original_page": pages[page_id]["original"]},
             "frame_indices": pages[page_id]["frame_indices"]}
            for page_id in page_ids]


def report(run, tasks, budget, pages):
    rows = read_rows(run)
    results = {}
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        results[method] = {
            "correct": sum(row["correct"] for row in group),
            "total": len(group),
            "episodes": {episode: sum(row["correct"] for row in group if row["episode"] == episode)
                         for episode in EPISODES},
            "mean_candidate_gold_coverage": (sum(row["candidate_gold_coverage"] for row in group) / len(group)
                                             if group else None),
            "empty_predictions": sum(not row["prediction"] for row in group),
        }
    output = {"complete": all(item["total"] == len(tasks) for item in results.values()),
              "results": results, "readagent_pages": {ep: len(pages.get(ep, [])) for ep in EPISODES},
              "api_usage": budget.state,
              "metric": "offline goal-frame hit, 12 development questions; not navigation SR/SPL"}
    save(run / "summary.json", output)
    lines = ["# 超图记忆与 ReadAgent 对比", "",
             "ReadAgent 直接执行固定官方 notebook 的分页与摘要函数；问题阶段调用其原有摘要回看提示词和页号解析。",
             "原 notebook 的最终答案是 QuALITY 选择题，当前任务改用双方共用的目标帧视觉回答器。",
             "共享两段视频共 192 帧的同一份已提取视觉观测；本次不重新抽帧或做视觉提取。",
             "ReadAgent 12 题重新作答；超图复用已用同样代码、输入及答题器生成的 12 题答案。",
             "使用相同的 Qwen 答题模型、提示、最多 12 张原图和 40KB 检索证据文本上限。",
             "ReadAgent 另外保留官方的整篇摘要，超图另外保留图规划结果；因此总文本量可能不同。",
             "超图方法采用 Top-K=20；ReadAgent 按其原方法选 1–6 页，检索单位不同，不强行统一页数与记录数。", "",
             "| 方法 | 合计 | 视频 1 | 视频 2 | 候选目标帧覆盖率 |", "|---|---:|---:|---:|---:|"]
    for method, value in results.items():
        coverage = value["mean_candidate_gold_coverage"]
        lines.append(f"| {method} | {value['correct']}/{value['total']} | "
                     f"{value['episodes']['ep_1']}/6 | {value['episodes']['ep_33']}/6 | "
                     f"{coverage:.1%} |" if coverage is not None else
                     f"| {method} | 0/0 | 0/6 | 0/6 | 未完成 |")
    lines += ["", f"ReadAgent 分页数：视频 1 为 {len(pages.get('ep_1', []))}，"
              f"视频 2 为 {len(pages.get('ep_33', []))}。",
              f"本次记录的 reported tokens：{budget.state['reported_tokens']:,}；"
              f"API 请求数：{budget.state['requests_started']}。",
              "分页与摘要的已完成 API 调用由上次中断运行的缓存复用，当前运行的 token 计数不包含它们。",
              "逐帧文字由原始共同视觉观测序列化而成；分页阶段没有问题或标准答案。",
              "这是一组已反复诊断的开发题；结果不构成泛化性能或官方 ReadAgent 论文任务复现。"]
    (run / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path,
                        default=Path("runtime/findingdory_run_20261013_graphpriority_amem_final"))
    parser.add_argument("--run-dir", type=Path,
                        default=Path("runtime/findingdory_run_20261013_readagent"))
    parser.add_argument("--data-dir", type=Path, default=Path("runtime/findingdory"))
    parser.add_argument("--hypergraph-answers-run", type=Path,
                        help="Reuse unchanged hypergraph answer rows from a complete paired run")
    args = parser.parse_args()
    source, run, data = args.source_run, args.run_dir, args.data_dir
    if run.resolve() in {source.resolve(), data.resolve()}:
        raise ValueError("Isolated output directory required")
    settings = Settings.load().model_copy(update={"timeout": 120, "retries": 0})
    settings.validate_live()
    prior = load(source / "evaluation_config.json")
    if {key: getattr(settings, key) for key in prior["models"]} != prior["models"]:
        raise ValueError("Current models differ from the shared source run")
    tasks = load(data / "selected_tasks.json")
    shared = {ep: load(source / "observations" / f"{ep}.json") for ep in EPISODES}
    if fingerprint(tasks) != prior["task_sha256"]:
        raise ValueError("Tasks changed")
    if any(fingerprint(shared[ep]) != prior["observation_sha256"][ep] for ep in EPISODES):
        raise ValueError("Shared observations changed")
    _, audit = official_source()
    roots = Path(__file__).parents[1]
    hashes = code_hashes() | {str(path.relative_to(roots)).replace("\\", "/"):
        hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            roots / "benchmarks" / "readagent_official_adapter.py",
            roots / "benchmarks" / "findingdory_visual_answer.py",
            roots / "benchmarks" / "answer_normalization.py",
            roots / "benchmarks" / "run_budget.py")}
    freeze(run / "evaluation_config.json", {
        "methods": list(METHODS), "source_run": str(source.resolve()),
        "models": prior["models"], "task_sha256": fingerprint(tasks),
        "observation_sha256": {ep: fingerprint(shared[ep]) for ep in EPISODES},
        "code_sha256": hashes, "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "readagent_official": audit, "answer_prompt_sha256": fingerprint(VISUAL_ANSWER_SYSTEM),
        "hypergraph_answers_reused_from": (str(args.hypergraph_answers_run.resolve())
                                            if args.hypergraph_answers_run else None),
        "readagent_pagination": {"word_limit": 600, "start_threshold": 280},
        "readagent_lookup_pages": "1-6", "hypergraph_top_k": 20,
        "answer_context_bytes": 40000, "answer_images": 12,
        "adaptations": ["robot frame paragraphs instead of Gutenberg paragraphs",
                        "budgeted Qwen call replaces notebook model provider",
                        "open question instead of QuALITY choice question",
                        "common visual answerer replaces QuALITY final choice scorer"]})
    budget = RunBudget(run, max_requests=500, max_reported_tokens=0)
    if args.hypergraph_answers_run and not (run / "predictions.jsonl").exists():
        prior_run = args.hypergraph_answers_run
        previous = load(prior_run / "evaluation_config.json")
        if (previous["models"] != prior["models"] or
                previous["task_sha256"] != fingerprint(tasks) or
                previous["observation_sha256"] != {ep: fingerprint(shared[ep]) for ep in EPISODES} or
                previous["answer_prompt_sha256"] != fingerprint(VISUAL_ANSWER_SYSTEM) or
                any(previous["code_sha256"].get(key) != value for key, value in code_hashes().items())):
            raise ValueError("Hypergraph answer source uses different code, model, tasks, observations, or prompt")
        reused = [row for row in read_rows(prior_run) if row["method"] == "hypergraph"]
        if len(reused) != len(tasks):
            raise ValueError("Hypergraph answer source is incomplete")
        with (run / "predictions.jsonl").open("w", encoding="utf-8") as output:
            for row in reused:
                output.write(json.dumps(row | {"reused_from": str(prior_run.resolve())},
                                        ensure_ascii=False) + "\n")
    for ep in EPISODES:
        target = run / "observations" / f"{ep}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source / "observations" / f"{ep}.json", target)
        if fingerprint(load(target)) != fingerprint(shared[ep]):
            raise ValueError("Local input changed")
        target = run / "memory" / f"{ep}_hypergraph.sqlite3"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source / "memory" / f"{ep}_hypergraph.sqlite3", target)
    records = {ep: structured_records(ep, raw["observations"]) for ep, raw in shared.items()}
    graphs = {ep: HypergraphMemory(run / "memory" / f"{ep}_hypergraph.sqlite3") for ep in EPISODES}
    for ep, graph in graphs.items():
        with graph.connection() as db:
            actual = {row[0] for row in db.execute("SELECT id FROM records")}
        if actual != {record.id for record in records[ep]}:
            raise ValueError("Copied hypergraph index does not match source observations")
    namespace = (settings.embedding_url or settings.llm_url).rstrip("/") + "|" + settings.embedding_model
    client = ComparisonClients(settings, budget)
    agent = OfficialReadAgent(client, run / "readagent" / "model_cache")
    pages = {}
    try:
        for ep in EPISODES:
            path = run / "readagent" / f"{ep}_pages.json"
            if path.exists():
                pages[ep] = load(path)
            else:
                pages[ep] = agent.build(ep, shared[ep]["observations"])
                save(path, pages[ep])
            print(json.dumps({"episode": ep, "readagent_pages": len(pages[ep]),
                              "tokens": budget.state["reported_tokens"]}), flush=True)
        done = {(row["row_idx"], row["method"]) for row in read_rows(run)}
        for task in tasks:
            ep = task["ep_id"]
            plan = parse_plan(task["question"])
            for method in METHODS:
                if (task["row_idx"], method) in done:
                    continue
                start = time.perf_counter()
                client.calls.clear()
                if method == "hypergraph":
                    query = history_query(task["question"], ep, records[ep])
                    query.recall_plan = plan
                    selected, retrieval = graphs[ep].search(
                        query, 20, strategy="hypergraph", embedder=client.embed,
                        embedding_namespace=namespace,
                        minimum_similarity=settings.semantic_min_similarity)
                    if retrieval["semantic_backend"] != "embedding":
                        raise ValueError("Hypergraph embedding fallback")
                    evidence = [{"id": record.id, "content": compact_record(record),
                                 "frame_indices": supported_frames([record])} for record in selected[:20]]
                    evidence, resolution = pack_evidence(
                        evidence, retrieval["plan_evidence"],
                        require_all_targets=plan.all_targets or bool(plan.visit_order))
                else:
                    page_ids, choice_text = agent.lookup(pages[ep], task["question"])
                    evidence = readagent_evidence(ep, pages[ep], page_ids)
                    resolution = agent.expanded_gist_memory(ep, pages[ep], page_ids)
                    evidence, _ = pack_evidence(evidence, max_bytes=40000)
                    if not evidence:
                        raise ValueError("No looked-up original page fits the shared context budget")
                    retrieval = {"page_ids": page_ids, "lookup_response": choice_text,
                                 "packed_page_ids": [int(item["id"].rsplit("_", 1)[1]) for item in evidence],
                                 "gist_pages": len(pages[ep]), "official_revision": audit["revision"]}
                answer = decode_visual(client, task["question"], evidence,
                                       shared[ep]["observations"], data, ep, resolution)
                gold = json.loads(task["answer"])
                available = {index for item in evidence for index in item["frame_indices"]}
                row = {"row_idx": task["row_idx"], "episode": ep, "method": method,
                       "question": task["question"], "prediction": answer["goal_frames"],
                       "gold": gold, "correct": score_frames(
                           answer["goal_frames"], gold,
                           task["low_level_category"] == "Unordered Revisitation"),
                       "answer": answer, "evidence": evidence, "retrieval": retrieval,
                       "packed_resolution": resolution, "calls": client.calls.copy(),
                       "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                       "candidate_gold_coverage": sum(bool(available & set(group)) for group in gold) / len(gold),
                       "candidate_frame_count": len(available),
                       "context_bytes": len(json.dumps({"evidence": evidence, "resolution": resolution},
                                                       ensure_ascii=False).encode())}
                with (run / "predictions.jsonl").open("a", encoding="utf-8") as output:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(json.dumps({"task": task["row_idx"], "method": method,
                                  "correct": row["correct"], "prediction": row["prediction"],
                                  "tokens": budget.state["reported_tokens"]}), flush=True)
    finally:
        client.close()
        summary = report(run, tasks, budget, pages)
        print(json.dumps({"complete": summary["complete"], "results": summary["results"],
                          "tokens": budget.state["reported_tokens"]}), flush=True)


if __name__ == "__main__":
    main()
