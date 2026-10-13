"""Rebuild memory without VLM calls or label-guided observation edits."""
import argparse
import hashlib
from pathlib import Path
import shutil

from memory_agent.settings import Settings
from memory_agent.memory import HypergraphMemory
from memory_agent.semantic import record_text
from benchmarks.findingdory_structured import build, freeze, fingerprint, load
from benchmarks.findingdory_memory import SCHEMA_VERSION
from benchmarks.run_budget import RunBudget


def prepare(source, run, data):
    if source.resolve() == run.resolve():
        raise ValueError("Use a new run; never modify the old perception or score")
    manifest = load(data / "manifest.json")
    sources = {ep: load(source / "observations" / f"{ep}.json") for ep in manifest["episodes"]}
    settings = Settings.load()
    config = {"representation_schema": SCHEMA_VERSION, "perception_source_run": str(source),
        "perception_configuration": load(source / "caption_config.json"),
        "source_observation_sha256": {ep:fingerprint(v) for ep,v in sources.items()},
        "memory_builder_sha256": hashlib.sha256(Path(__file__).with_name("findingdory_memory.py").read_bytes()).hexdigest(),
        "observations_rewritten": False, "new_visual_api_requests": 0}
    freeze(run / "refinement_config.json", config)
    run.mkdir(parents=True, exist_ok=True)
    for ep in sources:
        target = run / "observations" / f"{ep}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "observations" / f"{ep}.json", target)
    shutil.copy2(source / "caption_config.json", run / "caption_config.json")
    RunBudget(run, max_requests=40, max_reported_tokens=100_000)
    build(settings, data, run)
    # Reuse vectors only when provider AND exact semantic text hash still match.
    reused = 0
    for ep in sources:
        for strategy in ("flat", "hypergraph"):
            original = HypergraphMemory(source / "memory" / f"{ep}_{strategy}.sqlite3")
            target = HypergraphMemory(run / "memory" / f"{ep}_{strategy}.sqlite3")
            with original.connection() as db:
                cache = db.execute("SELECT record_id,provider,content_hash,vector_json FROM embedding_cache").fetchall()
            with target.connection(write=True) as db:
                for row in cache:
                    try:
                        current = target.get(row[0])
                    except KeyError:
                        continue
                    digest = hashlib.sha256(record_text(current).encode()).hexdigest()
                    if digest == row[2]:
                        db.execute("INSERT OR REPLACE INTO embedding_cache VALUES (?,?,?,?)", tuple(row))
                        reused += 1
    print({"cached_vectors_reused":reused,"new_visual_api_requests":0}, flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=Path("runtime/findingdory_run_20261012_qwen_v4"))
    parser.add_argument("--run-dir", type=Path, default=Path("runtime/findingdory_run_20261012_qwen_v5"))
    parser.add_argument("--data-dir", type=Path, default=Path("runtime/findingdory"))
    args=parser.parse_args()
    prepare(args.source_run,args.run_dir,args.data_dir)


if __name__=="__main__":
    main()
