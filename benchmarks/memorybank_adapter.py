"""Pinned official MemoryBank retrieval, adapted to witnessed video observations.

Execute upstream prompt builders, JSON loader and FAISS neighbor expansion.
Only legacy dependency/API plumbing and frame provenance are adapted. The
official default non-forgetting path is used; personality analysis is omitted
because a robot's visual observations are not a user's personality dialogue.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import List, Optional, Tuple

from benchmarks.comparison_methods import MemoryMethod, register
from benchmarks.comparison_worker import atomic_json
from benchmarks.findingdory_mem0 import RETRIEVAL_K
from benchmarks.findingdory_pilot import load
from benchmarks.findingdory_structured import fingerprint
from memory_agent.clients import ModelError
from memory_agent.recall import pack_evidence
from memory_agent.semantic import validate_vectors

REPOSITORY = "https://github.com/zhongwanjun/MemoryBank-SiliconFriend"
REVISION = "cf61c4196e4cfdb0f2b7a0316249fa40312dc3a9"
ROOT = Path(__file__).resolve().parents[1] / "runtime/vendor/memorybank"
SOURCE_HASHES = {
    "memory_bank/summarize_memory.py": "092b7a8d6b675216e7f2933f3a2290937d9de4a6b913f920ece62b764f28cb9f",
    "memory_bank/memory_retrieval/local_doc_qa.py": "40a1d76b790028e5576fdd5e169d8ce18c2bfac86fee890c3cfa95ea4c2cc756",
    "memory_bank/memory_retrieval/configs/model_config.py": "10e56e1105c3c8177e61b56c0f22d43730eff89e741d3cdd1d02cffb0f0b0cc5",
    "utils/sys_args.py": "dbadd93366c7b77ffeb15fef3c68f9993eb46136e1d21ae3b3a7ae87891139d7",
}
DAY = "historical_day"  # Clip-local logical day, not an invented calendar date.
CHUNK_SIZE = 200  # Upstream model_config.py, unchanged.


def official_source(root=None):
    root = Path(root) if root is not None else ROOT
    if not (root / ".git").exists():
        raise FileNotFoundError(f"MemoryBank 官方源码缺失：{root}；请克隆 {REPOSITORY} 并固定到 {REVISION}")
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise RuntimeError("MemoryBank 官方源码版本改变；停止实验以防混用结果")
    for name, digest in SOURCE_HASHES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"MemoryBank 官方文件被修改：{name}")
    return root, {"repository": REPOSITORY, "revision": REVISION, "source_sha256": dict(SOURCE_HASHES)}


class WitnessText(str):
    """Track exact input documents through upstream's in-place concatenation."""
    def __new__(cls, value, source_ids=()):
        instance = super().__new__(cls, value)
        instance.source_ids = tuple(source_ids)
        return instance

    def __add__(self, other):
        return WitnessText(str(self) + str(other), tuple(dict.fromkeys(
            (*self.source_ids, *getattr(other, "source_ids", ())))))

    def __reduce__(self):
        return WitnessText, (str(self), self.source_ids)


@dataclass
class Document:
    page_content: str
    metadata: dict = field(default_factory=dict)


class FileLoader:
    def __init__(self, filepath, **kwargs):
        self.file_path = filepath


def upstream_namespace(root):
    """Compile named upstream definitions, never its demos or legacy imports."""
    import numpy as np
    namespace = {"Document": Document, "UnstructuredFileLoader": FileLoader,
        "List": List, "Optional": Optional, "Tuple": Tuple, "TextSplitter": object,
        "json": json, "np": np, "print": lambda *args, **kwargs: None}
    wanted = {
        "memory_bank/summarize_memory.py": {"summarize_content_prompt"},
        "memory_bank/memory_retrieval/local_doc_qa.py": {
            "JsonMemoryLoader", "seperate_list", "similarity_search_with_score_by_vector"},
    }
    for name, definitions in wanted.items():
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name in definitions]
        if {node.name for node in nodes} != definitions:
            raise RuntimeError("Pinned MemoryBank definitions are missing")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root / name), "exec"), namespace)
    return namespace


class OfficialMemoryBank:
    def __init__(self, client):
        root, self.audit = official_source()
        self.namespace = upstream_namespace(root)
        self.client = client

    def summarize(self, history, episode):
        prompt = self.namespace["summarize_content_prompt"](history, episode, "Observer", "en")
        model = self.client.settings.llm_model
        response = self.client.post(self.client.settings.llm_url.rstrip("/") + "/chat/completions",
            self.client.settings.llm_key,
            {"model": model, "temperature": 0, "max_tokens": 400, **self.client.chat_options(model),
             "messages": [
                 {"role": "system", "content": "Below is a transcript of a conversation between a human and an AI assistant that is intelligent and knowledgeable in psychology."},
                 {"role": "user", "content": "Hello! Please help me summarize the content of the conversation."},
                 {"role": "system", "content": "Sure, I will do my best to assist you."},
                 {"role": "user", "content": prompt}]}, "memorybank_summary")
        try:
            content = response["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError()
            return content
        except (KeyError, IndexError, TypeError, ValueError):
            raise ModelError("memorybank_summary: empty or invalid response") from None

    def documents(self, memory_file, episode, indices):
        docs = self.namespace["JsonMemoryLoader"](str(memory_file), "en").load(episode)
        if len(docs) != len(indices) + 1:
            raise RuntimeError("Official MemoryBank loader changed document accounting")
        provenance = {}
        for index, doc in enumerate(docs):
            is_summary = index == len(indices)
            identity = f"{episode}_summary" if is_summary else f"{episode}_frame_{indices[index]}"
            doc.page_content = WitnessText(doc.page_content, [identity])
            # A day summary is not permission to expose all 96 original images.
            # Only independently retrieved raw frame documents expose images.
            provenance[identity] = [] if is_summary else [indices[index]]
        return docs, provenance

    def search(self, index, documents, embedding):
        # Upstream mutates doc.page_content during merging. Isolate each query
        # so question order/resumption cannot contaminate future retrievals.
        docs = {i: copy.deepcopy(doc) for i, doc in enumerate(documents)}
        store = SimpleNamespace(index=index, index_to_docstore_id={i: i for i in docs},
            docstore=SimpleNamespace(search=lambda identity: docs[identity]), chunk_size=CHUNK_SIZE)
        return self.namespace["similarity_search_with_score_by_vector"](store, embedding, RETRIEVAL_K)


@register
class MemoryBankMethod(MemoryMethod):
    name = "memorybank"
    environment = str(Path(__file__).resolve().parents[1] / "runtime/memorybank-env/Scripts/python.exe")

    def build(self):
        import faiss
        import numpy as np
        self.agent = OfficialMemoryBank(self.client)
        self.banks = {}
        for ep, frames in self.observations.items():
            indices = [frame["frame_index"] for frame in frames]
            if indices != list(range(len(frames))) or not frames:
                raise ValueError("MemoryBank requires chronological, nonempty shared observations")
            directory = self.run / "memorybank" / ep
            source_hash = fingerprint(frames)
            summary_path = directory / "summary_checkpoint.json"
            self.progress(stage="memory", episode=ep, completed=0, total=len(frames), phase="summary")
            history = [{"query": f"Observed frame {frame['frame_index']} at {frame['clock']}.",
                        "response": json.dumps(frame, ensure_ascii=False, sort_keys=True)} for frame in frames]
            if summary_path.exists():
                saved = load(summary_path)
                if saved["observations_sha256"] != source_hash:
                    raise RuntimeError("MemoryBank summary checkpoint has different observations")
                summary = saved["summary"]
            else:
                summary = self.agent.summarize(history, ep)
                atomic_json(summary_path, {"observations_sha256": source_hash, "summary": summary})
            bank = {ep: {"name": ep, "history": {DAY: history}, "summary": {DAY: {"content": summary}}}}
            memory_file = directory / "memory_bank.json"
            if memory_file.exists():
                if load(memory_file) != bank:
                    raise RuntimeError("MemoryBank persisted history differs from sealed observations")
            else:
                atomic_json(memory_file, bank)
            docs, provenance = self.agent.documents(memory_file, ep, indices)
            texts = [str(doc.page_content) for doc in docs]
            signature = fingerprint({"texts": texts, "embedding_model": self.client.settings.embedding_model,
                "embedding_url": self.client.settings.embedding_url or self.client.settings.llm_url})
            vector_file = directory / "embedding_checkpoint.json"
            saved = load(vector_file) if vector_file.exists() else {"fingerprint": signature, "vectors": []}
            if saved["fingerprint"] != signature or len(saved["vectors"]) > len(docs):
                raise RuntimeError("MemoryBank embedding checkpoint is incompatible")
            vectors = saved["vectors"]
            if vectors:
                validate_vectors(vectors, len(vectors))
            for start in range(len(vectors), len(texts), 16):
                vectors.extend(self.client.embed(texts[start:start + 16]))
                validate_vectors(vectors, len(vectors))
                atomic_json(vector_file, {"fingerprint": signature, "vectors": vectors})
                self.progress(stage="memory", episode=ep, completed=min(len(vectors), len(frames)),
                              total=len(frames), phase="embedding")
            array = np.asarray(validate_vectors(vectors, len(docs)), dtype=np.float32)
            index = faiss.IndexFlatL2(array.shape[1])
            index.add(array)
            self.banks[ep] = (index, docs, provenance)
            self.progress(stage="memory", episode=ep, completed=len(frames), total=len(frames), phase="ready")

    def retrieve(self, episode, question):
        embedding = self.client.embed([question])[0]
        index, documents, provenance = self.banks[episode]
        result = self.agent.search(index, documents, embedding)
        evidence = []
        sources = {}
        for position, (document, score) in enumerate(result):
            ids = document.page_content.source_ids
            if not ids or any(identity not in provenance for identity in ids):
                raise RuntimeError("MemoryBank retrieval lost its original document provenance")
            identity = f"{episode}_memorybank_{position}_" + hashlib.sha256("|".join(ids).encode()).hexdigest()[:12]
            evidence.append({"id": identity, "content": str(document.page_content),
                "frame_indices": sorted({n for source in ids for n in provenance[source]})})
            # The legacy helper reuses its last seed score; treat it only as
            # diagnostics, and serialize padding infinity as null on tiny tests.
            sources[identity] = {"document_ids": list(ids),
                "upstream_group_score": float(score) if math.isfinite(float(score)) else None}
        evidence, _ = pack_evidence(evidence)
        if not evidence:
            raise RuntimeError("MemoryBank returned no evidence within the common 40KB budget")
        return evidence, None, {"official_revision": REVISION, "retrieval_top_k": RETRIEVAL_K,
            "neighbor_chunk_size": CHUNK_SIZE, "forgetting_enabled": False, "personality_enabled": False,
            "summary_has_no_eligible_images": True,
            "sources": {item["id"]: sources[item["id"]] for item in evidence}}
