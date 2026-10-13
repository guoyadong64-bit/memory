"""No-network memory diagnostics from saved observations, never answer accuracy.

Task labels are used only after retrieval for diagnostic target coverage. Queries,
memory construction, and evidence selection never receive reference answers.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from contextlib import contextmanager
import shutil
from uuid import uuid4

from benchmarks.findingdory_memory import structured_records, history_query, supported_frames, SCHEMA_VERSION
from benchmarks.findingdory_mem0 import ANSWER_SYSTEM, RETRIEVAL_K, answer_request, code_hashes
from memory_agent.memory import HypergraphMemory
from memory_agent.recall import compact_record, graph_plan, pack_evidence, parse_plan


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def request_fingerprint(question, evidence, resolution):
    return fingerprint({"system":ANSWER_SYSTEM,"request":answer_request(question,evidence,resolution)})


@contextmanager
def replay_workspace(parent):
    # Normal mkdir inherits workspace ACLs on Windows; tempfile's private ACLs
    # can exclude the process identity used by the desktop sandbox.
    path = parent / ("memory-replay-" + uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        resolved = path.resolve()
        if resolved.parent != parent.resolve() or not resolved.name.startswith("memory-replay-"):
            raise ValueError("Refusing to clean outside the replay workspace")
        shutil.rmtree(resolved)


def replay(source, data, output):
    source, data, output = source.resolve(), data.resolve(), output.resolve()
    if output == source or source in output.parents or output == data or data in output.parents:
        raise ValueError("Replay output must not overwrite the source experiment or dataset")
    tasks = read(data / "selected_tasks.json")
    previous = [json.loads(line) for line in (source / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    old = {(row["row_idx"],row["method"]):row for row in previous}
    raw = {ep:read(source / "observations" / f"{ep}.json")["observations"]
           for ep in read(data / "manifest.json")["episodes"]}
    old_config = read(source / "evaluation_config.json")
    old_system_matches = old_config["answer_prompt_sha256"] == fingerprint(ANSWER_SYSTEM)
    rows, diagnostics = [], {}
    # Only this temporary subtree is removed; source memories are never opened for writing.
    output.parent.mkdir(parents=True, exist_ok=True)
    with replay_workspace(output.parent) as temporary:
        for ep, observations in raw.items():
            records = structured_records(ep, observations)
            store = HypergraphMemory(Path(temporary) / f"{ep}.sqlite3")
            store.add_many(records)
            processes = [r for r in records if r.evolution]
            diagnostics[ep] = {"frames":len(observations), "processes":len(processes),
                "contextual_labels":sum(len(e.labels) for r in records if not r.evolution for e in r.entities),
                "locally_confirmed_pickup_processes":sum(any(s.state.get("pickup_evidence") == "direct_lift"
                                                            for s in r.evolution.stages) for r in processes)}
            for task in (task for task in tasks if task["ep_id"] == ep):
                question = task["question"]
                query = history_query(question, ep, records)
                query.recall_plan = parse_plan(question)
                selected, retrieval = store.search(query, RETRIEVAL_K, strategy="hypergraph")
                evidence = [{"id":record.id,"content":compact_record(record),
                             "frame_indices":supported_frames([record])} for record in selected[:RETRIEVAL_K]]
                evidence, resolution = pack_evidence(evidence, retrieval.get("plan_evidence"),
                    require_all_targets=query.recall_plan.all_targets or bool(query.recall_plan.visit_order))
                prior = old.get((task["row_idx"],"hypergraph"))
                current_hash = request_fingerprint(question, evidence, resolution)
                prior_hash = request_fingerprint(question, prior["evidence"], prior["retrieval"].get("plan_evidence")) if prior else None
                unchanged = bool(prior and old_system_matches and current_hash == prior_hash)
                # Diagnostic only; reference labels are introduced AFTER all memory operations.
                gold = json.loads(task["answer"])
                candidates = {n for item in evidence for n in item["frame_indices"]}
                target_frames = {target["frame_index"] for target in resolution.get("targets", [])}
                rows.append({"task_id":task["row_idx"],"episode":ep,"question":question,
                    "diagnostic_backend":"lexical; not equivalent to the live embedding experiment",
                    "resolution_complete":resolution["complete"],"warnings":resolution["warnings"],
                    "targets":resolution["targets"],
                    "prior_resolution_complete":prior["retrieval"]["plan_evidence"]["complete"] if prior else None,
                    "candidate_goal_coverage":sum(bool(candidates & set(group)) for group in gold)/len(gold),
                    "resolved_target_goal_coverage":sum(bool(target_frames & set(group)) for group in gold)/len(gold),
                    "context_bytes":len(json.dumps(evidence,ensure_ascii=False).encode()),
                    "new_answer_request_sha256":current_hash,"prior_answer_request_sha256":prior_hash,
                    "request_unchanged":unchanged,"requires_live_answer_evaluation":not unchanged})
            diagnostics[ep]["sqlite_integrity"] = "ok"
            with store.connection() as db:
                if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise RuntimeError("Replay memory integrity check failed")
    baseline_checks = []
    for prior in previous:
        if prior["method"] != "mem0":
            continue
        evidence, _ = pack_evidence(prior["evidence"])
        unchanged = old_system_matches and request_fingerprint(prior["question"],evidence,None) == request_fingerprint(
            prior["question"],prior["evidence"],None)
        baseline_checks.append({"task_id":prior["row_idx"],"existing_request_unchanged":unchanged,
            "scope":"decoder/context only; retrieval, observations and model config must also remain unchanged"})
    result = {"kind":"offline memory diagnostic; NOT a new answer accuracy result", "api_calls":0,
        "schema":SCHEMA_VERSION,"source_run":str(source),"source_observation_sha256":{ep:fingerprint(v) for ep,v in raw.items()},
        "code_sha256":code_hashes(), "diagnostics":diagnostics,
        "summary":{"queries":len(rows),"resolved_queries":sum(row["resolution_complete"] for row in rows),
            "prior_resolved_queries":sum(bool(row["prior_resolution_complete"]) for row in rows),
            "changed_answer_requests":sum(not row["request_unchanged"] for row in rows),
            "warnings":dict(Counter(w for row in rows for w in row["warnings"]))},
        "baseline_dependency_checks":baseline_checks,"queries":rows}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run",type=Path,default=Path("runtime/findingdory_run_20261012_mem0_v9"))
    parser.add_argument("--data-dir",type=Path,default=Path("runtime/findingdory"))
    parser.add_argument("--output",type=Path,default=Path("runtime/findingdory_replay_v10.json"))
    args = parser.parse_args()
    result = replay(args.source_run,args.data_dir,args.output)
    print(json.dumps({"api_calls":0,"summary":result["summary"],"diagnostics":result["diagnostics"],
                      "output":str(args.output.resolve())},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
