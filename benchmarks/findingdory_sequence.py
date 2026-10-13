"""Continuous-frame perception repair and a fair fresh-memory comparison.

Only the selected episode's images are reinterpreted. No task text or reference
answers enter the visual extraction. Both memories receive identical new facts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from benchmarks.findingdory_memory import SCHEMA_VERSION, history_query, structured_records, supported_frames, validate_batch
from benchmarks.findingdory_mem0 import (ANSWER_SYSTEM, RETRIEVAL_K, ComparisonClients, code_hashes, decode, read_rows)
from benchmarks.findingdory_pilot import load, save, score_frames
from benchmarks.findingdory_structured import CAPTION_SYSTEM, extract_batch, fingerprint, freeze, persist_records
from benchmarks.mem0_adapter import create_memory, observation_message, memory_frame_indices
from benchmarks.run_budget import RunBudget
from memory_agent.memory import HypergraphMemory
from memory_agent.recall import compact_record, pack_evidence, parse_plan
from memory_agent.settings import Settings
from benchmarks.findingdory_visual_answer import VISUAL_ANSWER_SYSTEM, decode_visual


SEQUENCE_SYSTEM = CAPTION_SYSTEM + """
Additional continuous-frame audit:
Interpret before/current/after images as a sequence. Frame descriptions and registry
names are hypotheses; the pictures remain the evidence. Never infer a lift from an
object's disappearance alone. If the current grip is occluded, keep uncertainty;
use adjacent images to identify the same physical item, without copying a relation
that cannot be seen in the requested frame.
The simulated robot's own body includes large white/grey cylinders, mechanical
blocks, screws, finger plates and black articulated links attached to the gripper.
These are parts of the actor, NOT external objects being picked up. Do not declare
a held_by edge from a robot component to itself, or call an arm component a lamp.
Inspect what lies BETWEEN the gripping fingers, and what changes support across
images. Do not mistake open-looking grey finger plates for an empty hand when an
external object can be seen between the black tips. Do not invent an invisible object.
Use physical appearance in new stable IDs/names, and category labels separately.
Flat rectangular electronic items are not automatically books or phones; read their
visible construction and retain an uncertain category when needed. Similar-shaped
objects at different locations are not automatically the same instance.
Distinguish an item's actual supporting surface from an overlapping background.
A woven top supporting an item is not necessarily the interior of a storage box.
Represent supporting furniture and movable items separately. Do not emit all scenery.
For each entity, attributes must include visibility (clear, partial, occluded) and
role (movable, support, fixture, robot_component). Emit the robot as an agent; its
integral components need not be separate entities. Avoid a high-confidence category
for an object only visible as a tiny fragment. Keep concise descriptions and only
the relations that support the observed interaction or location.
"""


def prepare(settings, source, baseline, data, run, reuse_observations=False):
    if run.resolve() in {source.resolve(),baseline.resolve(),data.resolve()}:
        raise ValueError("Use an isolated new run")
    prior = load(source/"api_budget.json")
    limit = prior["limits"]["max_reported_tokens"]
    models = {key:getattr(settings,key) for key in ("llm_model","vision_model","answer_model","embedding_model")}
    if models != load(source/"evaluation_config.json")["models"]:
        raise ValueError("Model settings changed; freeze a separate experiment")
    config = {"schema":SCHEMA_VERSION,"models":models,
        "source_run":str(source.resolve()),"baseline_source":str(baseline.resolve()),
        "task_sha256":fingerprint(load(data/"selected_tasks.json")),"methods":["hypergraph","mem0"],
        "episode_refined":"ep_33","code_sha256":code_hashes(),
        "driver_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "visual_prompt_sha256":fingerprint(SEQUENCE_SYSTEM),"answer_prompt_sha256":fingerprint(VISUAL_ANSWER_SYSTEM),
        "answer_mode":"common evidence-bounded images, maximum 12 per question",
        "visual_input":"images only, no task text or answers","same_new_observations_for_both_methods":True,
        "retrieval_k":RETRIEVAL_K,"max_context_utf8_bytes":40000,"max_reported_tokens":limit}
    config_path = run/"evaluation_config.json"
    if config_path.exists():
        old_config = load(config_path)
        ignored = {"fingerprint","code_sha256","driver_sha256"}
        if {k:v for k,v in old_config.items() if k not in ignored} != {k:v for k,v in config.items() if k not in ignored}:
            raise ValueError("Experimental inputs/protocol changed; use another run")
        if old_config["code_sha256"] != config["code_sha256"] or old_config["driver_sha256"] != config["driver_sha256"]:
            if (run/"memory").exists():
                raise ValueError("Memory has already been built with other code; use another run")
            # Perception checkpoints are immutable. Capture the representation code
            # used for its FIRST build, rather than relabelling an existing memory.
            freeze(run/"implementation_revisions"/(fingerprint(config)[:12]+".json"),config)
    else:
        freeze(config_path,config)
    budget = RunBudget(run,max_requests=300,max_reported_tokens=limit)
    if "carried_usage" not in budget.state:
        budget.state["reported_tokens"] = prior["reported_tokens"]
        budget.state["carried_usage"] = {"source_run":str(source.resolve()),"reported_tokens":prior["reported_tokens"]}
        budget.state["historical_usage"] = prior.get("historical_usage",{})
        budget.save()
    for ep in (("ep_1","ep_33") if reuse_observations else ("ep_1",)):
        destination = run/"observations"/f"{ep}.json"
        destination.parent.mkdir(parents=True,exist_ok=True)
        original = source/"observations"/f"{ep}.json"
        if destination.exists() and fingerprint(load(destination)) != fingerprint(load(original)):
            raise ValueError("Reused episode observations changed")
        if not destination.exists():
            shutil.copy2(original,destination)
    freeze(run/"caption_config.json",{"schema":SCHEMA_VERSION,"vision_model":settings.vision_model or settings.llm_model,
        "prompt_sha256":fingerprint(SEQUENCE_SYSTEM),"target_frames_per_call":4,
        "context_before":2,"context_after":2,"reused_episode":"ep_1","refined_episode":"ep_33",
        "input_questions":False,"input_reference_answers":False,"max_output_tokens":8000,
        "identity_policy":"fresh registry, descriptive stable names, evidence-bearing categories",
        "reuse_source_observations":reuse_observations})
    return budget


def caption(settings, data, run, budget):
    completed = run/"observations"/"ep_33.json"
    if completed.exists():
        return
    previous = []
    client = ComparisonClients(settings,budget)
    try:
        for start in range(0,96,4):
            indices = list(range(start,start+4))
            cache = run/"caption_batches"/"ep_33"/f"{start:03d}.json"
            if cache.exists():
                batch = validate_batch(load(cache),indices,previous)
            else:
                freeze(run/"caption_microbatch_config.json",{"target_frames_per_call":2,
                    "context_before":2,"context_after":2,"max_output_tokens":6000,
                    "transport_failure_policy":"one bounded retry of this microbatch; no completed batch repeats"})
                batch = []
                for substart in (start,start+2):
                    requested = [substart,substart+1]
                    microcache = run/"caption_microbatches"/f"{substart:03d}.json"
                    if microcache.exists():
                        subbatch = validate_batch(load(microcache),requested,[*previous,*batch])
                    else:
                        from memory_agent.clients import ModelError
                        for attempt in range(2):
                            try:
                                subbatch = extract_batch(client,data,"ep_33",requested,[*previous,*batch],
                                    run/"validation_checks"/f"{substart:03d}.json",system=SEQUENCE_SYSTEM,
                                    context_before=2,context_after=2,max_output_tokens=6000,max_attempts=2)
                                break
                            except ModelError as exc:
                                if str(exc) != "visual_observation: network request failed" or attempt:
                                    raise
                                print(json.dumps({"retry_microbatch":requested,"reason":"network timeout"}),flush=True)
                        save(microcache,[f.model_dump(mode="json") for f in subbatch])
                    batch.extend(subbatch)
                save(cache,[f.model_dump(mode="json") for f in batch])
            previous.extend(batch)
            print(json.dumps({"caption_frames_done":len(previous),"tokens":budget.state["reported_tokens"]}),flush=True)
        save(completed,{"episode":"ep_33","schema_version":SCHEMA_VERSION,
            "observations":[f.model_dump(mode="json") for f in previous],
            "provenance":"Label-blind continuous-frame VLM extraction; model hypotheses, not ground truth"})
    finally:
        client.close()


def compare(settings, source, baseline, data, run, budget, fresh_all_indexes=False):
    records = {ep:structured_records(ep,load(run/"observations"/f"{ep}.json")["observations"])
               for ep in ("ep_1","ep_33")}
    graphs = {ep:persist_records(run,ep,rows,"hypergraph") for ep, rows in records.items()}
    dims = load(baseline/"embedding_dimensions.json")["dimensions"]
    freeze(run/"mem0_config.json",{k:v for k,v in load(baseline/"mem0_config.json").items() if k != "fingerprint"} | {
        "new_input_ep_33_sha256":fingerprint(load(run/"observations"/"ep_33.json")),
        "ep_1":("fresh official SDK extraction and indexing" if fresh_all_indexes else "unchanged SDK index copied; all answers regenerated"),
        "ep_33":"fresh official SDK extraction and indexing"})
    done = {(row["row_idx"],row["method"]) for row in read_rows(run)}
    namespace = (settings.embedding_url or settings.llm_url).rstrip("/")+"|"+settings.embedding_model
    client = ComparisonClients(settings,budget)
    memories = {}
    tasks = load(data/"selected_tasks.json")
    try:
        # The common decoder changed: regenerate ALL answers. Only the unchanged
        # ep_1 SDK index is copied; no previous answer is used as a new prediction.
        old_index = baseline/"mem0"/"ep_1"
        new_index = run/"mem0"/"ep_1"
        if not fresh_all_indexes and not new_index.exists():
            shutil.copytree(old_index,new_index)
        memories["ep_1"] = create_memory(new_index,client,dims)
        memories["ep_33"] = create_memory(run/"mem0"/"ep_33",client,dims)
        for indexed_ep in (("ep_1","ep_33") if fresh_all_indexes else ("ep_33",)):
            memory = memories[indexed_ep]
            observations = load(run/"observations"/f"{indexed_ep}.json")["observations"]
            for start in range(0,len(observations),8):
                path = run/"mem0_ingestion"/indexed_ep/f"{start:03d}.json" if fresh_all_indexes else run/"mem0_ingestion"/f"{start:03d}.json"
                if not path.exists():
                    subset = observations[start:start+8]
                    result = memory.add([observation_message(f) for f in subset],user_id=indexed_ep,
                        metadata={"episode":indexed_ep,"frame_indices":[f["frame_index"] for f in subset]},infer=True)
                    save(path,result)
                print(json.dumps({"mem0_episode":indexed_ep,"mem0_frames_indexed":min(start+8,len(observations)),
                    "tokens":budget.state["reported_tokens"]}),flush=True)
            snapshot = memory.get_all(filters={"user_id":indexed_ep},top_k=10000)
            if not snapshot.get("results"):
                raise RuntimeError("Mem0 extracted no facts; cannot report an empty baseline")
            save(run/"mem0_facts"/f"{indexed_ep}.json",snapshot)
        for task in tasks:
            ep = task["ep_id"]
            for method in ("hypergraph","mem0"):
                if (task["row_idx"],method) in done:
                    continue
                start = time.perf_counter();client.calls.clear()
                plan = parse_plan(task["question"])
                if method == "hypergraph":
                    query = history_query(task["question"],ep,records[ep]);query.recall_plan = plan
                    selected,retrieval = graphs[ep].search(query,RETRIEVAL_K,strategy="hypergraph",embedder=client.embed,
                        embedding_namespace=namespace,minimum_similarity=settings.semantic_min_similarity)
                    if retrieval["semantic_backend"] != "embedding":
                        raise RuntimeError("Embedding fallback; stopped instead of mixing backends")
                    evidence = [{"id":r.id,"content":compact_record(r),"frame_indices":supported_frames([r])}
                                for r in selected[:RETRIEVAL_K]]
                    resolution = retrieval.get("plan_evidence")
                else:
                    retrieval = memories[ep].search(task["question"],top_k=RETRIEVAL_K,filters={"user_id":ep},explain=True)
                    evidence = [{"id":m["id"],"content":m["memory"],"frame_indices":memory_frame_indices(m)}
                                for m in retrieval["results"]]
                    resolution = None
                evidence,resolution = pack_evidence(evidence,resolution,
                    require_all_targets=plan.all_targets or bool(plan.visit_order))
                shared = load(run/"observations"/f"{ep}.json")["observations"]
                answer = decode_visual(client,task["question"],evidence,shared,data,ep,resolution)
                gold = json.loads(task["answer"])
                candidate_frames = {n for item in evidence for n in item["frame_indices"]}
                row = {"row_idx":task["row_idx"],"episode":ep,"method":method,"question":task["question"],
                    "prediction":answer["goal_frames"],"gold":gold,"correct":score_frames(answer["goal_frames"],gold,
                    task["low_level_category"] == "Unordered Revisitation"),"answer":answer,"evidence":evidence,
                    "retrieval":retrieval,"packed_resolution":resolution,"calls":client.calls.copy(),"reused":False,
                    "latency_ms":round((time.perf_counter()-start)*1000,2),
                    "candidate_gold_coverage":sum(bool(candidate_frames & set(g)) for g in gold)/len(gold)}
                with (run/"predictions.jsonl").open("a",encoding="utf-8") as file:
                    file.write(json.dumps(row,ensure_ascii=False)+"\n")
                print(json.dumps({"task":task["row_idx"],"method":method,"correct":row["correct"],
                    "prediction":row["prediction"],"tokens":budget.state["reported_tokens"]}),flush=True)
    finally:
        for memory in memories.values():memory.close()
        client.close()
        from benchmarks.findingdory_retest import report
        summary = report(data,run)
        print(json.dumps({"complete":summary["complete"],"results":summary["results"],
            "total_current_budget_tokens":budget.state["reported_tokens"]}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run",type=Path,default=Path("runtime/findingdory_run_20261012_mem0_v10"))
    parser.add_argument("--baseline-run",type=Path,default=Path("runtime/findingdory_run_20261012_mem0_v9"))
    parser.add_argument("--run-dir",type=Path,default=Path("runtime/findingdory_run_20261012_mem0_v11"))
    parser.add_argument("--data-dir",type=Path,default=Path("runtime/findingdory"))
    parser.add_argument("--stage",choices=("caption","evaluate","all"),default="all")
    parser.add_argument("--reuse-observations",action="store_true")
    args = parser.parse_args()
    settings = Settings.load().model_copy(update={"timeout":120,"retries":0})
    budget = prepare(settings,args.source_run,args.baseline_run,args.data_dir,args.run_dir,args.reuse_observations)
    if args.stage in {"caption","all"}:caption(settings,args.data_dir,args.run_dir,budget)
    if args.stage in {"evaluate","all"}:compare(settings,args.source_run,args.baseline_run,args.data_dir,args.run_dir,budget)


if __name__ == "__main__":main()
