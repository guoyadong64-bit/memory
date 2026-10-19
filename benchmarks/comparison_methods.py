"""Memory-method registry for the common FindingDory comparison runner.

Each method owns ingestion and retrieval. The worker owns answering and scoring.
Third-party methods may register a subclass via --plugin module.path.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.findingdory_memory import frame_times, history_query, structured_records, supported_frames, validate_batch
from benchmarks.findingdory_mem0 import RETRIEVAL_K
from benchmarks.findingdory_pilot import load
from memory_agent.recall import compact_record, pack_evidence, parse_plan


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


REGISTRY = {}


def register(cls):
    if not cls.name or cls.name in REGISTRY:
        raise ValueError(f"Duplicate or empty comparison method: {cls.name}")
    REGISTRY[cls.name] = cls
    return cls


class MemoryMethod:
    name = ""
    environment = "core"  # core, mem0, amem; a plugin may name its own Python executable

    def __init__(self, run: Path, data: Path, observations: dict, client, progress):
        self.run, self.data, self.observations = run, data, observations
        self.client, self.progress = client, progress

    def build(self):
        raise NotImplementedError

    def retrieve(self, episode: str, question: str):
        """Return (evidence list, resolution or None, retrieval diagnostics)."""
        raise NotImplementedError

    def close(self):
        pass


@register
class HypergraphMethod(MemoryMethod):
    name = "hypergraph"

    def build(self):
        from memory_agent.memory import HypergraphMemory
        self.records, self.graphs = {}, {}
        for ep, frames in self.observations.items():
            records = structured_records(ep, frames)
            graph = HypergraphMemory(self.run / "memory" / f"{ep}.sqlite3")
            with graph.connection() as db:
                found = {row[0] for row in db.execute("SELECT id FROM records")}
            expected = {record.id for record in records}
            if not found:
                graph.add_many(records)
            elif found != expected:
                raise ValueError(f"Partial hypergraph index for {ep}; use a new run directory")
            self.records[ep], self.graphs[ep] = records, graph
            self.progress(stage="memory", episode=ep, completed=len(records), total=len(records))

    def retrieve(self, episode, question):
        settings = self.client.settings
        namespace = (settings.embedding_url or settings.llm_url).rstrip("/") + "|" + settings.embedding_model
        query = history_query(question, episode, self.records[episode])
        plan = parse_plan(question)
        query.recall_plan = plan
        selected, retrieval = self.graphs[episode].search(
            query, RETRIEVAL_K, strategy="hypergraph", embedder=self.client.embed,
            embedding_namespace=namespace, minimum_similarity=settings.semantic_min_similarity)
        if retrieval["semantic_backend"] != "embedding":
            raise ValueError("Hypergraph embedding fallback; comparison stopped")
        evidence = [{"id": record.id, "content": compact_record(record),
                     "frame_indices": supported_frames([record])} for record in selected[:RETRIEVAL_K]]
        evidence, resolution = pack_evidence(
            evidence, retrieval["plan_evidence"],
            require_all_targets=plan.all_targets or bool(plan.visit_order))
        return evidence, resolution, retrieval


@register
class Mem0Method(MemoryMethod):
    name = "mem0"
    environment = "mem0"

    def build(self):
        from benchmarks.mem0_adapter import MEM0_VERSION, create_memory, observation_message
        from importlib.metadata import version
        if version("mem0ai") != MEM0_VERSION:
            raise RuntimeError("Mem0 SDK version changed")
        dim_file = self.run / "embedding_dimensions.json"
        if dim_file.exists():
            dimensions = load(dim_file)["dimensions"]
        else:
            dimensions = len(self.client.embed(["robot observation"])[0])
            atomic_json(dim_file, {"model": self.client.settings.embedding_model, "dimensions": dimensions})
        self.memories = {}
        for ep, frames in self.observations.items():
            memory = create_memory(self.run / "mem0" / ep, self.client, dimensions)
            self.memories[ep] = memory
            for start in range(0, len(frames), 8):
                checkpoint = self.run / "mem0_ingestion" / ep / f"{start:03d}.json"
                if not checkpoint.exists():
                    subset = frames[start:start + 8]
                    result = memory.add(
                        [observation_message(frame) for frame in subset], user_id=ep,
                        metadata={"episode": ep, "frame_indices": [f["frame_index"] for f in subset]}, infer=True)
                    atomic_json(checkpoint, result)
                self.progress(stage="memory", episode=ep, completed=min(start + 8, len(frames)), total=len(frames))
            snapshot = memory.get_all(filters={"user_id": ep}, top_k=10000)
            if not snapshot.get("results"):
                raise RuntimeError("Mem0 generated no facts")
            atomic_json(self.run / "mem0_facts" / f"{ep}.json", snapshot)

    def retrieve(self, episode, question):
        from benchmarks.mem0_adapter import memory_frame_indices
        result = self.memories[episode].search(question, top_k=RETRIEVAL_K,
                                               filters={"user_id": episode}, explain=True)
        evidence = [{"id": item["id"], "content": item["memory"],
                     "frame_indices": memory_frame_indices(item)} for item in result["results"]]
        evidence, _ = pack_evidence(evidence)
        return evidence, None, result

    def close(self):
        for memory in getattr(self, "memories", {}).values():
            memory.close()


@register
class AMemMethod(MemoryMethod):
    name = "amem"
    environment = "amem"

    def build(self):
        from benchmarks.amem_adapter import create_memory, save_state, observation_content
        self.memories, self.provenance = {}, {}
        for ep, frames in self.observations.items():
            directory = self.run / "amem" / ep
            memory = create_memory(directory, self.client)
            self.memories[ep] = memory
            times = frame_times(validate_batch(frames, list(range(len(frames)))))
            self.provenance[ep] = {f"{ep}_frame_{frame['frame_index']}": [frame["frame_index"]]
                                   for frame in frames}
            for count, frame in enumerate(frames, 1):
                identity = f"{ep}_frame_{frame['frame_index']}"
                content = observation_content(frame)
                if identity in memory.memories:
                    if memory.read(identity).content != content:
                        raise ValueError("A-Mem resumed with different original observation")
                else:
                    memory.add_note(content, id=identity,
                                    time=times[frame["frame_index"]][0].strftime("%Y%m%d%H%M"))
                    note = memory.read(identity)
                    if not note.keywords or not note.tags or note.context == "General":
                        raise ValueError("A-Mem metadata extraction degraded")
                    save_state(memory, directory)
                if count % 8 == 0 or count == len(frames):
                    self.progress(stage="memory", episode=ep, completed=count, total=len(frames))

    def retrieve(self, episode, question):
        from benchmarks.amem_adapter import evidence as amem_evidence, REVISION
        result = self.memories[episode].search_agentic(question, k=RETRIEVAL_K)
        if not result:
            raise ValueError("Official A-Mem search returned no result")
        evidence = amem_evidence(self.memories[episode], result, self.provenance[episode])
        evidence, _ = pack_evidence(evidence)
        return evidence, None, {"results": result, "official_method": "search_agentic", "revision": REVISION}


@register
class ReadAgentMethod(MemoryMethod):
    name = "readagent"

    def build(self):
        from benchmarks.readagent_official_adapter import OfficialReadAgent
        self.agent = OfficialReadAgent(self.client, self.run / "readagent" / "model_cache")
        self.pages = {}
        for ep, frames in self.observations.items():
            path = self.run / "readagent" / f"{ep}_pages.json"
            self.pages[ep] = load(path) if path.exists() else self.agent.build(ep, frames)
            if not path.exists():
                atomic_json(path, self.pages[ep])
            self.progress(stage="memory", episode=ep, completed=len(frames), total=len(frames),
                          pages=len(self.pages[ep]))

    def retrieve(self, episode, question):
        from benchmarks.findingdory_readagent import readagent_evidence
        pages = self.pages[episode]
        page_ids, response = self.agent.lookup(pages, question)
        evidence = readagent_evidence(episode, pages, page_ids)
        resolution = self.agent.expanded_gist_memory(episode, pages, page_ids)
        evidence, _ = pack_evidence(evidence)
        if not evidence:
            raise ValueError("No ReadAgent original page fits the evidence budget")
        return evidence, resolution, {
            "page_ids": page_ids, "packed_page_ids": [int(x["id"].rsplit("_", 1)[1]) for x in evidence],
            "lookup_response": response, "gist_pages": len(pages),
            "official_revision": self.agent.audit["revision"]}
