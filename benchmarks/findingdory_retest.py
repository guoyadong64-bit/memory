"""Paid retest of the current hypergraph with unchanged Mem0 answers reused.

No visual extraction or Mem0 indexing is performed. A fresh budget is created
only for a new experiment, retaining the previous usage as historical metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time

from benchmarks.findingdory_memory import structured_records, history_query, supported_frames, SCHEMA_VERSION
from benchmarks.findingdory_mem0 import (ANSWER_SYSTEM, RETRIEVAL_K, ComparisonClients,
    answer_request, code_hashes, decode, read_rows)
from benchmarks.findingdory_pilot import load, save, score_frames
from benchmarks.findingdory_structured import fingerprint, freeze, persist_records
from benchmarks.run_budget import RunBudget
from memory_agent.recall import compact_record, pack_evidence, parse_plan
from memory_agent.semantic import record_text
from memory_agent.settings import Settings


def prepare(settings, source, data, run, token_limit):
    if run.resolve() in {source.resolve(),data.resolve()}:
        raise ValueError("Use a new experiment directory")
    old_config = load(source/"evaluation_config.json")
    current_models = {k:getattr(settings,k) for k in old_config["models"]}
    if current_models != old_config["models"] or fingerprint(ANSWER_SYSTEM) != old_config["answer_prompt_sha256"]:
        raise ValueError("Models or common prompt changed; cannot reuse the Mem0 baseline")
    hashes = code_hashes()
    if hashes["benchmarks/mem0_adapter.py"] != old_config["code_sha256"]["benchmarks/mem0_adapter.py"]:
        raise ValueError("Mem0 adapter changed; cannot reuse baseline")
    tasks = load(data/"selected_tasks.json")
    if fingerprint(tasks) != old_config["task_sha256"] or old_config["retrieval_k"] != RETRIEVAL_K:
        raise ValueError("Tasks or recall depth changed; cannot reuse baseline")
    previous_rows = read_rows(source)
    baseline = [row for row in previous_rows if row["method"] == "mem0"]
    if sorted(row["row_idx"] for row in baseline) != sorted(task["row_idx"] for task in tasks):
        raise ValueError("Baseline must contain exactly one result per task")
    for row in baseline:
        packed, _ = pack_evidence(row["evidence"])
        if answer_request(row["question"],packed,None) != answer_request(row["question"],row["evidence"],None):
            raise ValueError("Mem0 answer context changed; cannot reuse baseline")
    old_budget = load(source/"api_budget.json")
    namespace = (settings.embedding_url or settings.llm_url).rstrip("/")+"|"+settings.embedding_model
    observation_hashes = {}
    built = {}
    for ep in load(data/"manifest.json")["episodes"]:
        raw = load(source/"observations"/f"{ep}.json")
        if fingerprint(raw) != old_config["observation_sha256"][ep]:
            raise ValueError("Shared observations changed; cannot reuse baseline")
        observation_hashes[ep] = fingerprint(raw)
        built[ep] = structured_records(ep,raw["observations"])
        with sqlite3.connect("file:"+(source/"memory"/f"{ep}_hypergraph.sqlite3").resolve().as_posix()+"?mode=ro",uri=True) as db:
            providers = {row[0] for row in db.execute("SELECT DISTINCT provider FROM embedding_cache")}
        if providers != {namespace}:
            raise ValueError("Embedding endpoint/model changed; cannot reuse baseline")
    config = {"schema":SCHEMA_VERSION,"source_run":str(source.resolve()),"models":current_models,
        "answer_prompt_sha256":fingerprint(ANSWER_SYSTEM),"task_sha256":fingerprint(tasks),
        "source_observation_sha256":observation_hashes,"code_sha256":hashes,
        "retest_driver_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "baseline_rows_sha256":fingerprint(baseline),"baseline_reuse":"unchanged inputs, models, adapter and answer requests",
        "max_reported_tokens":token_limit,"fresh_budget_authorized":True,
        "previous_reported_tokens":old_budget["reported_tokens"],"new_visual_api_calls":0,
        "retrieval_k":RETRIEVAL_K,"max_context_utf8_bytes":40000,"max_answer_output_tokens":2048}
    freeze(run/"evaluation_config.json",config)
    budget = RunBudget(run,max_requests=100,max_reported_tokens=token_limit)
    if "historical_usage" not in budget.state:
        budget.state["historical_usage"] = {"source_run":str(source.resolve()),
            "reported_tokens":old_budget["reported_tokens"],"reason":"User authorized a fresh 5M-token experiment; past usage remains recorded"}
        budget.save()
    done = {(row["row_idx"],row["method"]) for row in read_rows(run)}
    with (run/"predictions.jsonl").open("a",encoding="utf-8") as file:
        for row in baseline:
            if (row["row_idx"],"mem0") not in done:
                copied = row | {"reused_from":str(source.resolve()),"reused":True,"calls":[],
                    "latency_note":"Historical baseline latency; no new Mem0 call"}
                file.write(json.dumps(copied,ensure_ascii=False)+"\n")
    reused_vectors = 0
    for ep, records in built.items():
        destination = run/"observations"/f"{ep}.json"
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():
            shutil.copy2(source/"observations"/f"{ep}.json",destination)
        elif fingerprint(load(destination)) != observation_hashes[ep]:
            raise ValueError("Saved retest observations changed")
        graph = persist_records(run,ep,records,"hypergraph")
        text_hashes = {r.id:hashlib.sha256(record_text(r).encode()).hexdigest() for r in records}
        with sqlite3.connect("file:"+(source/"memory"/f"{ep}_hypergraph.sqlite3").resolve().as_posix()+"?mode=ro",uri=True) as db:
            cache = db.execute("SELECT record_id,provider,content_hash,vector_json FROM embedding_cache").fetchall()
        with graph.connection(write=True) as db:
            for row in cache:
                if text_hashes.get(row[0]) == row[2] and row[1] == namespace:
                    db.execute("INSERT OR IGNORE INTO embedding_cache VALUES (?,?,?,?)",row)
                    reused_vectors += 1
    return built, budget, reused_vectors


def report(data, run):
    rows = read_rows(run)
    n = len(load(data/"selected_tasks.json"))
    results = {}
    for method in ("hypergraph","mem0"):
        values = [r for r in rows if r["method"] == method]
        results[method] = {"correct":sum(r["correct"] for r in values),"total":len(values),
            "accuracy":sum(r["correct"] for r in values)/len(values) if values else None,
            "empty_predictions":sum(not r["prediction"] for r in values),
            "correct_task_ids":[r["row_idx"] for r in values if r["correct"]],
            "reused_queries":sum(r.get("reused",False) for r in values)}
    budget = load(run/"api_budget.json")
    previous = budget["historical_usage"]["reported_tokens"]
    carried = budget.get("carried_usage",{}).get("reported_tokens",0)
    summary = {"complete":all(r["total"] == n for r in results.values()),
        "metric":"offline goal-frame membership; 12 development questions, not navigation SR/SPL",
        "results":results,"budget":budget,"this_experiment_reported_tokens":budget["reported_tokens"],
        "new_stage_reported_tokens":budget["reported_tokens"]-carried,
        "historical_reported_tokens":previous,"all_experiments_reported_tokens":previous+budget["reported_tokens"]}
    save(run/"summary.json",summary)
    reused = results["mem0"]["reused_queries"]
    text = ["# 新版超图记忆重新作答结果", "", "双方使用同一份视觉观测及相同回答模型、提示词。",
        f"Mem0 的 {reused} 个未变化输入结果复用，其余结果重新生成；具体视觉提取与索引配置见 evaluation_config.json。", "",
        "| 方法 | 正确/总数 | 命中率 |", "|---|---:|---:|"]
    for method, values in results.items():
        accuracy = f"{values['accuracy']:.1%}" if values["accuracy"] is not None else "未完成"
        text.append(f"| {method} | {values['correct']}/{values['total']} | {accuracy} |")
    limit_label = f"{budget['limits']['max_reported_tokens']:,}" if budget['limits']['max_reported_tokens'] else "无累计上限"
    text.extend(["",f"当前额度累计 token：{budget['reported_tokens']:,}/{limit_label}；本阶段新增：{budget['reported_tokens']-carried:,}；历史 token：{previous:,}。",
        "评分要求目标数和顺序符合题目，并命中标准目标帧区间。开发题已参与调试，不支持泛化优越性结论。",
        "Mem0 基线是原先的官方 OSS 场景适配配置，spaCy/BM25 和图存储未启用。",
        ("复用结果的延迟来自原实验，不能当成本轮重新测量值。" if reused else
         "双方答案均重新生成。检索耗时可能包含首次嵌入更新，不能直接用来证明速度优势。")])
    (run/"REPORT.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    return summary


def evaluate(settings, source, data, run, token_limit):
    built, budget, cached = prepare(settings,source,data,run,token_limit)
    print(json.dumps({"prepared":True,"reused_embedding_vectors":cached,"budget_tokens":budget.state["reported_tokens"],
                      "models":{"answer":settings.answer_model or settings.llm_model,"embedding":settings.embedding_model}}),flush=True)
    done = {r["row_idx"] for r in read_rows(run) if r["method"] == "hypergraph"}
    client = ComparisonClients(settings,budget)
    namespace = (settings.embedding_url or settings.llm_url).rstrip("/")+"|"+settings.embedding_model
    tasks = load(data/"selected_tasks.json")
    try:
        from memory_agent.memory import HypergraphMemory
        for ep, records in built.items():
            graph = HypergraphMemory(run/"memory"/f"{ep}_hypergraph.sqlite3")
            for task in (t for t in tasks if t["ep_id"] == ep and t["row_idx"] not in done):
                start = time.perf_counter()
                client.calls.clear()
                query = history_query(task["question"],ep,records)
                query.recall_plan = parse_plan(task["question"])
                candidates, retrieval = graph.search(query,RETRIEVAL_K,strategy="hypergraph",
                    embedder=client.embed,embedding_namespace=namespace,minimum_similarity=settings.semantic_min_similarity)
                if retrieval["semantic_backend"] != "embedding":
                    raise RuntimeError("Embedding failed; stopped instead of mixing retrieval backends")
                evidence = [{"id":r.id,"content":compact_record(r),"frame_indices":supported_frames([r])}
                            for r in candidates[:RETRIEVAL_K]]
                evidence,resolution = pack_evidence(evidence,retrieval.get("plan_evidence"),
                    require_all_targets=query.recall_plan.all_targets or bool(query.recall_plan.visit_order))
                answer = decode(client,task["question"],evidence,resolution)
                # Reference answers are used only by this evaluator after the model finishes.
                gold = json.loads(task["answer"])
                candidate_frames = {n for item in evidence for n in item["frame_indices"]}
                row = {"row_idx":task["row_idx"],"episode":ep,"method":"hypergraph","question":task["question"],
                    "category":task["high_level_category"],"prediction":answer["goal_frames"],"gold":gold,
                    "correct":score_frames(answer["goal_frames"],gold,task["low_level_category"] == "Unordered Revisitation"),
                    "answer":answer,"evidence":evidence,"retrieval":retrieval,"packed_resolution":resolution,
                    "latency_ms":round((time.perf_counter()-start)*1000,2),"calls":client.calls.copy(),"reused":False,
                    "candidate_gold_coverage":sum(bool(candidate_frames & set(g)) for g in gold)/len(gold),
                    "candidate_frame_count":len(candidate_frames),"context_bytes":len(json.dumps(evidence,ensure_ascii=False).encode())}
                with (run/"predictions.jsonl").open("a",encoding="utf-8") as file:
                    file.write(json.dumps(row,ensure_ascii=False)+"\n")
                print(json.dumps({"task":task["row_idx"],"episode":ep,"correct":row["correct"],
                    "prediction":row["prediction"],"tokens_so_far":budget.state["reported_tokens"]}),flush=True)
    finally:
        client.close()
        summary = report(data,run)
        print(json.dumps({"summary":summary["results"],"complete":summary["complete"],
                          "new_reported_tokens":summary["this_experiment_reported_tokens"]}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run",type=Path,default=Path("runtime/findingdory_run_20261012_mem0_v9"))
    parser.add_argument("--run-dir",type=Path,default=Path("runtime/findingdory_run_20261012_mem0_v10"))
    parser.add_argument("--data-dir",type=Path,default=Path("runtime/findingdory"))
    parser.add_argument("--max-tokens",type=int,default=5_000_000)
    args = parser.parse_args()
    settings = Settings.load().model_copy(update={"timeout":90,"retries":0})
    if not settings.llm_key or not settings.embedding_model:
        raise ValueError("Configure the answer API key and embedding model before retesting")
    evaluate(settings,args.source_run,args.data_dir,args.run_dir,args.max_tokens)


if __name__ == "__main__":
    main()
