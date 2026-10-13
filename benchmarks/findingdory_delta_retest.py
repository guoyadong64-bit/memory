"""Regenerate only changed hypergraph answer requests; no answer-based selection."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from benchmarks.findingdory_memory import structured_records, history_query, supported_frames
from benchmarks.findingdory_mem0 import ComparisonClients, answer_request, code_hashes, read_rows
from benchmarks.findingdory_pilot import load, save, score_frames
from benchmarks.findingdory_retest import report
from benchmarks.findingdory_structured import fingerprint, freeze
from benchmarks.findingdory_visual_answer import VISUAL_ANSWER_SYSTEM, decode_visual, select_images
from benchmarks.run_budget import RunBudget
from memory_agent.memory import HypergraphMemory
from memory_agent.recall import compact_record, pack_evidence, parse_plan
from memory_agent.settings import Settings


def request_hash(question, evidence, observations, resolution):
    images = select_images(evidence,observations,resolution)
    return fingerprint({"system":VISUAL_ANSWER_SYSTEM,"request":answer_request(question,evidence,resolution),
                        "visual_frames":images})


def evaluate(settings, source, data, run):
    if run.resolve() == source.resolve():raise ValueError("Use another run")
    old_config=load(source/"evaluation_config.json")
    models={k:getattr(settings,k) for k in old_config["models"]}
    if models != old_config["models"] or fingerprint(VISUAL_ANSWER_SYSTEM) != old_config["answer_prompt_sha256"]:
        raise ValueError("Common model/prompt changed; cannot reuse answers")
    if code_hashes()["benchmarks/mem0_adapter.py"] != old_config["code_sha256"]["benchmarks/mem0_adapter.py"]:
        raise ValueError("Mem0 adapter changed; rerun the baseline")
    tasks=load(data/"selected_tasks.json")
    if fingerprint(tasks) != old_config["task_sha256"]:raise ValueError("Tasks changed")
    rows=read_rows(source)
    old={(r["row_idx"],r["method"]):r for r in rows}
    if len(old)!=24:raise ValueError("Require a completed source comparison")
    shared={ep:load(source/"observations"/f"{ep}.json")["observations"] for ep in ("ep_1","ep_33")}
    for r in rows:
        if r["method"] == "mem0" and select_images(r["evidence"],shared[r["episode"]],None) != r["answer"]["visual_frames"]:
            raise ValueError("Baseline visual request changed")
    config={"models":models,"source_run":str(source.resolve()),"answer_prompt_sha256":fingerprint(VISUAL_ANSWER_SYSTEM),
        "code_sha256":code_hashes(),"driver_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "observation_sha256":{ep:fingerprint(raw) for ep,raw in shared.items()},"task_sha256":fingerprint(tasks),
        "selection_policy":"request hash only, never correctness", "retrieval_k":20,"image_limit":12,
        "baseline":"all 12 unchanged-input results reused"}
    freeze(run/"evaluation_config.json",config)
    previous=load(source/"api_budget.json")
    budget=RunBudget(run,max_requests=80,max_reported_tokens=previous["limits"]["max_reported_tokens"])
    if "carried_usage" not in budget.state:
        budget.state["reported_tokens"]=previous["reported_tokens"]
        budget.state["carried_usage"]={"source_run":str(source.resolve()),"reported_tokens":previous["reported_tokens"]}
        budget.state["historical_usage"]=previous["historical_usage"]
        budget.save()
    for ep in shared:
        destination=run/"observations"/f"{ep}.json";destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():shutil.copy2(source/"observations"/f"{ep}.json",destination)
        destination=run/"memory"/f"{ep}_hypergraph.sqlite3";destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():shutil.copy2(source/"memory"/f"{ep}_hypergraph.sqlite3",destination)
    done={(r["row_idx"],r["method"]) for r in read_rows(run)}
    with (run/"predictions.jsonl").open("a",encoding="utf-8") as file:
        for r in rows:
            if r["method"] == "mem0" and (r["row_idx"],"mem0") not in done:
                file.write(json.dumps(r|{"calls":[],"reused":True,"reused_from":str(source.resolve())},ensure_ascii=False)+"\n")
    records={ep:structured_records(ep,raw) for ep,raw in shared.items()}
    graphs={ep:HypergraphMemory(run/"memory"/f"{ep}_hypergraph.sqlite3") for ep in shared}
    namespace=(settings.embedding_url or settings.llm_url).rstrip("/")+"|"+settings.embedding_model
    for graph in graphs.values():
        with graph.connection() as db:
            if {r[0] for r in db.execute("SELECT DISTINCT provider FROM embedding_cache")} != {namespace}:
                raise ValueError("Embedding endpoint/model changed")
    client=ComparisonClients(settings,budget)
    changes=[]
    try:
        for task in tasks:
            if (task["row_idx"],"hypergraph") in done:continue
            ep=task["ep_id"];query=history_query(task["question"],ep,records[ep]);query.recall_plan=parse_plan(task["question"])
            client.calls.clear();start=time.perf_counter()
            selected,retrieval=graphs[ep].search(query,20,strategy="hypergraph",embedder=client.embed,
                embedding_namespace=namespace,minimum_similarity=settings.semantic_min_similarity)
            if retrieval["semantic_backend"] != "embedding":raise RuntimeError("Embedding fallback")
            evidence=[{"id":r.id,"content":compact_record(r),"frame_indices":supported_frames([r])} for r in selected[:20]]
            evidence,resolution=pack_evidence(evidence,retrieval["plan_evidence"],
                require_all_targets=query.recall_plan.all_targets or bool(query.recall_plan.visit_order))
            prior=old[(task["row_idx"],"hypergraph")]
            current_hash=request_hash(task["question"],evidence,shared[ep],resolution)
            old_hash=request_hash(task["question"],prior["evidence"],shared[ep],prior["packed_resolution"])
            unchanged=current_hash == old_hash
            changes.append({"task_id":task["row_idx"],"old_request_sha256":old_hash,
                            "new_request_sha256":current_hash,"unchanged":unchanged})
            if unchanged:
                row=prior|{"reused":True,"reused_from":str(source.resolve()),"calls":client.calls.copy()}
            else:
                answer=decode_visual(client,task["question"],evidence,shared[ep],data,ep,resolution)
                gold=json.loads(task["answer"])
                candidate_frames={n for item in evidence for n in item["frame_indices"]}
                row={"row_idx":task["row_idx"],"episode":ep,"method":"hypergraph","question":task["question"],
                    "prediction":answer["goal_frames"],"gold":gold,"correct":score_frames(answer["goal_frames"],gold,
                    task["low_level_category"]=="Unordered Revisitation"),"answer":answer,"evidence":evidence,
                    "retrieval":retrieval,"packed_resolution":resolution,"reused":False,"calls":client.calls.copy(),
                    "latency_ms":round((time.perf_counter()-start)*1000,2),
                    "candidate_gold_coverage":sum(bool(candidate_frames&set(g)) for g in gold)/len(gold)}
            with (run/"predictions.jsonl").open("a",encoding="utf-8") as file:
                file.write(json.dumps(row,ensure_ascii=False)+"\n")
            print(json.dumps({"task":task["row_idx"],"reused":unchanged,"correct":row["correct"],
                              "prediction":row["prediction"],"tokens":budget.state["reported_tokens"]}),flush=True)
    finally:
        client.close();save(run/"request_dependency_checks.json",changes)
        summary=report(data,run)
        print(json.dumps({"complete":summary["complete"],"results":summary["results"],
                          "tokens":budget.state["reported_tokens"]}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run",type=Path,default=Path("runtime/findingdory_run_20261012_mem0_v12"))
    parser.add_argument("--run-dir",type=Path,default=Path("runtime/findingdory_run_20261012_mem0_v13"))
    parser.add_argument("--data-dir",type=Path,default=Path("runtime/findingdory"))
    args=parser.parse_args()
    evaluate(Settings.load().model_copy(update={"timeout":120,"retries":0}),args.source_run,args.data_dir,args.run_dir)


if __name__ == "__main__":main()
