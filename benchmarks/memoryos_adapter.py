"""Run pinned MemoryOS classes with cached API transport and atomic checkpoints.

Upstream memory algorithms and prompts remain unchanged. This adapter replaces
local embedding downloads, SDK transport, random IDs, wall clocks and fragmented
file writes; it does not use the upstream chatbot or write evaluation QA back.
"""
from __future__ import annotations

import ast
from collections import defaultdict, deque
import copy
from datetime import datetime, timedelta
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
from types import ModuleType
from uuid import uuid4

from benchmarks.comparison_methods import MemoryMethod, register
from benchmarks.comparison_worker import atomic_json
from benchmarks.findingdory_pilot import load
from benchmarks.findingdory_structured import fingerprint
from memory_agent.clients import ModelError
from memory_agent.recall import pack_evidence

REPOSITORY = "https://github.com/BAI-LAB/MemoryOS"
REVISION = "587ed7755c7aed179965792830ff1b5ad9a6fa92"
ROOT = Path(__file__).resolve().parents[1] / "runtime/vendor/memoryos"
SOURCE_HASHES = {
    "__init__.py": "fa85dd7381478ac6d8b87cb63ccce893e8b11955307036612111c90f82cea096",
    "long_term.py": "9a8c85a1644af23d658d52668605af3a8a3e5cfe8eddb8fe4975390dafd76830",
    "memoryos.py": "fadf77b8ae52ac9fddaafa2a83b19580d1e1c4e1cdd1debf863ed07ea783e544",
    "mid_term.py": "f7fdab732c1a419f5245f21c3493d417c387b7f22b8f35e78341625b7e926ec8",
    "prompts.py": "07b92e08d85ac43925bbef993c252be8b51cb76a16254f3c9d9c0080daaae310",
    "retriever.py": "2c7ccecda2d6764e2d3c199c2a0f3779e556173dddf58123be58e1083db90de9",
    "short_term.py": "1d7be4c641a44efd65610cf4b85bd5b4c9daf4bba84f1390e2b56bd0f511d2ec",
    "updater.py": "f0da3ebe114b063a91ae61bf507e0b4d2de768242ded5c87ceb0bfb7429a7041",
    "utils.py": "29e1ba2b4a1718e31abb7dbbc89914e9a1c7f91445f33a4e427f630b4f47b060",
}
PARAMETERS = {"short_term_capacity": 10, "mid_term_capacity": 2000,
    "long_term_knowledge_capacity": 100, "retrieval_queue_capacity": 7,
    "mid_term_heat_threshold": 5.0, "mid_term_similarity_threshold": .6}


def official_source(root=None):
    root = Path(root) if root is not None else ROOT
    if not (root / ".git").exists():
        raise FileNotFoundError(f"MemoryOS 官方源码缺失：{root}；请克隆 {REPOSITORY} 并固定到 {REVISION}")
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise RuntimeError("MemoryOS 官方源码提交改变")
    for name, digest in SOURCE_HASHES.items():
        if hashlib.sha256((root / "memoryos-pypi" / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"MemoryOS 官方文件被修改：{name}")
    return root / "memoryos-pypi", {"repository": REPOSITORY, "revision": REVISION,
                                  "variant": "memoryos-pypi", "source_sha256": dict(SOURCE_HASHES)}


def observation_pair(episode, frame):
    return {"user_input": f"Historical visual observation {episode}/frame/{frame['frame_index']}.",
            "agent_response": json.dumps(frame, ensure_ascii=False, sort_keys=True)}


class OfficialMemoryOS:
    """Each instance owns private upstream module globals and one video bank."""
    def __init__(self, client, episode, directory, progress=None):
        self.client, self.episode, self.directory = client, episode, Path(directory)
        self.progress = progress or (lambda **event: None)
        self.lock = threading.RLock()
        self.counter, self.position, self.total = 0, 0, 0
        self.clock = "2000-01-01 00:00:00"
        self.failure = None
        self.embeddings = {}
        root, self.audit = official_source()
        self.package = f"_findingdory_memoryos_{uuid4().hex}"
        parent = ModuleType(self.package)
        parent.__path__ = [str(root)]
        sys.modules[self.package] = parent
        self.modules = {}
        try:
            prompts = self._module(root, "prompts")
            import numpy as np
            from concurrent.futures import ThreadPoolExecutor, as_completed
            utils = ModuleType(self.package + ".utils")
            utils.__package__ = self.package
            utils.__dict__.update(np=np, json=json, inspect=inspect, threading=threading,
                os=__import__("os"), prompts=prompts, print=lambda *a, **k: None,
                ThreadPoolExecutor=ThreadPoolExecutor, as_completed=as_completed,
                OpenAIClient=lambda **kwargs: self,
                get_embedding=self.embed, get_timestamp=lambda: self.clock,
                generate_id=self.generate_id, _embedding_cache={})
            tree = ast.parse((root / "utils.py").read_text(encoding="utf-8"))
            excluded = {"get_embedding", "get_timestamp", "generate_id"}
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name not in excluded]
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(root / "utils.py"), "exec"), utils.__dict__)
            sys.modules[utils.__name__] = utils
            self.modules["utils"] = utils
            self.prompts = prompts
            for name in ("short_term", "mid_term", "long_term", "updater", "retriever", "memoryos"):
                self._module(root, name)
        except BaseException:
            self.close()
            raise

    def _module(self, root, name):
        spec = importlib.util.spec_from_file_location(self.package + "." + name, root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        module.print = lambda *a, **k: None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.modules[name] = module
        return module

    def generate_id(self, prefix="id"):
        self.counter += 1
        digest = hashlib.sha256(f"{self.episode}:{self.counter}".encode()).hexdigest()[:12]
        return prefix + "_" + digest

    def raise_if_failed(self):
        if self.failure is not None:
            raise ModelError("memoryos: upstream caught a model failure; uncommitted state discarded") from self.failure

    def embed(self, text, **kwargs):
        import numpy as np
        with self.lock:
            try:
                if text not in self.embeddings:
                    self.embeddings[text] = self.client.embed([text])[0]
                return np.asarray(self.embeddings[text], dtype=np.float32)
            except Exception as exc:
                self.failure = exc
                raise

    def chat_completion(self, model, messages, temperature=.7, max_tokens=2000):
        system = messages[0]["content"]
        roles = {
            self.prompts.CONTINUITY_CHECK_SYSTEM_PROMPT: "memoryos_continuity",
            self.prompts.META_INFO_SYSTEM_PROMPT: "memoryos_chain_summary",
            self.prompts.MULTI_SUMMARY_SYSTEM_PROMPT: "memoryos_topic_summary",
            self.prompts.PERSONALITY_ANALYSIS_SYSTEM_PROMPT: "memoryos_profile",
            self.prompts.KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT: "memoryos_knowledge",
        }
        if system not in roles:
            raise RuntimeError("Unexpected MemoryOS generation stage; chatbot generation is disabled")
        role = roles[system]
        with self.lock:
            try:
                self.progress(stage="memory", episode=self.episode, completed=self.position,
                              total=self.total, phase=role)
                # Same LLM as the other methods; no upstream hidden SDK retries.
                for attempt in range(2):
                    request_messages = copy.deepcopy(messages)
                    if attempt:
                        request_messages.append({"role": "user", "content":
                            "Return only the required JSON array of themes or literal true/false as instructed. Do not use Markdown fences."})
                    response = self.client.post(self.client.settings.llm_url.rstrip("/") + "/chat/completions",
                        self.client.settings.llm_key,
                        {"model": self.client.settings.llm_model, "temperature": 0, "max_tokens": max_tokens,
                         **self.client.chat_options(self.client.settings.llm_model),
                         "messages": request_messages}, role)
                    try:
                        text = response["choices"][0]["message"]["content"]
                        if not isinstance(text, str) or not text.strip():
                            raise ValueError()
                        text = self.modules["utils"].clean_reasoning_model_output(text)
                        if role == "memoryos_topic_summary":
                            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
                            themes = json.loads(text)
                            if (not isinstance(themes, list) or len(themes) > 2 or any(
                                not isinstance(item, dict) or not isinstance(item.get("content"), str)
                                or not item["content"].strip() or not isinstance(item.get("keywords", []), list)
                                or any(not isinstance(word, str) for word in item.get("keywords", [])) for item in themes)):
                                raise ValueError()
                        if role == "memoryos_continuity" and text.lower() not in {"true", "false"}:
                            raise ValueError()
                        return text
                    except (KeyError, IndexError, TypeError, ValueError):
                        if attempt:
                            raise ModelError(f"{role}: invalid structured model response") from None
                raise AssertionError("unreachable")
            except Exception as exc:
                self.failure = exc
                raise

    def new_memory(self, state=None):
        self.failure = None
        if state is not None:
            self.counter = state["id_counter"]
            self.position = state["processed"]
            self.clock = state["logical_clock"]
        memory = self.modules["memoryos"].Memoryos(user_id=self.episode, assistant_id=self.episode,
            openai_api_key="adapter-transport", openai_base_url="https://adapter.invalid",
            data_storage_path=str(self.directory / "official_work"),
            llm_model=self.client.settings.llm_model if self.client else "preflight",
            embedding_model_name=self.client.settings.embedding_model if self.client else "preflight",
            **PARAMETERS)
        # A whole-bank commit replaces upstream's non-transactional file saves.
        for layer in (memory.short_term_memory, memory.mid_term_memory,
                      memory.user_long_term_memory, memory.assistant_long_term_memory):
            layer.save = lambda: None
        if state is not None:
            memory.short_term_memory.memory = deque(copy.deepcopy(state["short_term"]), maxlen=10)
            memory.mid_term_memory.sessions = copy.deepcopy(state["mid_term"]["sessions"])
            memory.mid_term_memory.access_frequency = defaultdict(int, state["mid_term"]["access_frequency"])
            memory.mid_term_memory.rebuild_heap()
            for key, layer in (("user_long_term", memory.user_long_term_memory),
                               ("assistant_long_term", memory.assistant_long_term_memory)):
                layer.user_profiles = copy.deepcopy(state[key]["user_profiles"])
                layer.knowledge_base = deque(copy.deepcopy(state[key]["knowledge_base"]), maxlen=100)
                layer.assistant_knowledge = deque(copy.deepcopy(state[key]["assistant_knowledge"]), maxlen=100)
            memory.updater.last_evicted_page_for_continuity = copy.deepcopy(state["last_evicted_page"])
        return memory

    def snapshot(self, memory, processed, source_hash):
        def long_term(layer):
            return {"user_profiles": layer.user_profiles, "knowledge_base": list(layer.knowledge_base),
                    "assistant_knowledge": list(layer.assistant_knowledge)}
        state = {"protocol": "memoryos-atomic-video-state-v1", "official_revision": REVISION,
            "observations_sha256": source_hash, "parameters": PARAMETERS,
            "processed": processed, "id_counter": self.counter, "logical_clock": self.clock,
            "short_term": memory.short_term_memory.get_all(),
            "mid_term": {"sessions": memory.mid_term_memory.sessions,
                         "access_frequency": dict(memory.mid_term_memory.access_frequency)},
            "user_long_term": long_term(memory.user_long_term_memory),
            "assistant_long_term": long_term(memory.assistant_long_term_memory),
            "last_evicted_page": memory.updater.last_evicted_page_for_continuity}
        import numpy as np
        def scalar(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value).__name__)
        return json.loads(json.dumps(state, default=scalar, allow_nan=False))

    def close(self):
        for name in list(sys.modules):
            if name == self.package or name.startswith(self.package + "."):
                del sys.modules[name]


@register
class MemoryOSMethod(MemoryMethod):
    name = "memoryos"
    environment = str(Path(__file__).resolve().parents[1] / "runtime/memoryos-env/Scripts/python.exe")

    def build(self):
        self.banks = {}
        for ep, frames in self.observations.items():
            if not frames or [f["frame_index"] for f in frames] != list(range(len(frames))):
                raise ValueError("MemoryOS requires nonempty chronological observations")
            directory = self.run / "memoryos" / ep
            engine = OfficialMemoryOS(self.client, ep, directory, self.progress)
            engine.total = len(frames)
            self.banks[ep] = (engine, None, frames)
            path, source_hash = directory / "state.json", fingerprint(frames)
            state = load(path) if path.exists() else None
            if state is not None and (state.get("observations_sha256") != source_hash
                or state.get("official_revision") != REVISION or state.get("parameters") != PARAMETERS
                or state.get("protocol") != "memoryos-atomic-video-state-v1"
                or type(state.get("processed")) is not int or not 0 <= state["processed"] <= len(frames)):
                raise RuntimeError("MemoryOS checkpoint does not match sealed observations or implementation")
            memory = engine.new_memory(state)
            start = state["processed"] if state else 0
            for n in range(start, len(frames)):
                engine.clock = (datetime(2000, 1, 1) + timedelta(seconds=n)).strftime("%Y-%m-%d %H:%M:%S")
                memory.add_memory(**observation_pair(ep, frames[n]), timestamp=engine.clock)
                engine.raise_if_failed()
                state = engine.snapshot(memory, n + 1, source_hash)
                atomic_json(path, state)
                engine.position = n + 1
                self.progress(stage="memory", episode=ep, completed=n + 1, total=len(frames), phase="ingest")
            self.banks[ep] = (engine, state, frames)
            self.progress(stage="memory", episode=ep, completed=len(frames), total=len(frames), phase="ready")

    def retrieve(self, episode, question):
        engine, state, frames = self.banks[episode]
        # Each evaluation query starts from the same frozen built bank. Upstream
        # search updates heat/visits, but those mutations are not committed.
        memory = engine.new_memory(state)
        engine.clock = (datetime(2000, 1, 1) + timedelta(seconds=len(frames))).strftime("%Y-%m-%d %H:%M:%S")
        result = memory.retriever.retrieve_context(user_query=question, user_id=episode)
        engine.raise_if_failed()
        evidence = []
        for key in ("retrieved_user_knowledge", "retrieved_assistant_knowledge"):
            if result[key]:
                evidence.append({"id": f"{episode}_memoryos_{key}", "frame_indices": [],
                    "content": {"layer": "long_term", "derived_hypotheses_not_image_proof": True,
                                "knowledge": [item["knowledge"] for item in result[key]]}})
        profile = memory.get_user_profile_summary()
        if profile and profile.strip().lower() != "none":
            evidence.append({"id": f"{episode}_memoryos_profile", "frame_indices": [],
                "content": {"layer": "long_term_profile", "derived_hypotheses_not_image_proof": True,
                            "profile": profile}})
        originals = {fingerprint(observation_pair(episode, frame)): frame for frame in frames}
        seen = set()
        # Budget priority: derived context, query-relevant pages, short-term.
        for layer, pages in (("mid_term", result["retrieved_pages"]),
                             ("short_term", memory.short_term_memory.get_all())):
            for page in pages:
                original = originals.get(fingerprint({key: page[key] for key in ("user_input", "agent_response")}))
                if original is None:
                    raise RuntimeError("MemoryOS recalled page without exact original observation provenance")
                n = original["frame_index"]
                if n in seen:
                    continue
                seen.add(n)
                evidence.append({"id": f"{episode}_memoryos_frame_{n}", "frame_indices": [n],
                    "content": {"layer": layer, "original_observation": original,
                        "chain_overview_hypothesis": page.get("meta_info"),
                        "source_page_id": page.get("page_id")}})
        if len(evidence) > 20:
            raise RuntimeError("MemoryOS context exceeded fixed record budget")
        evidence, _ = pack_evidence(evidence)
        return evidence, None, {"official_revision": REVISION, "parameters": PARAMETERS,
            "retrieval_thresholds": {"segment": .1, "page": .1, "knowledge": .01, "top_k_sessions": 5},
            "short_term_count": len(memory.short_term_memory.get_all()),
            "mid_term_pages_recalled": len(result["retrieved_pages"]),
            "long_term_knowledge_recalled": sum(len(result[key]) for key in
                ("retrieved_user_knowledge", "retrieved_assistant_knowledge")),
            "derived_context_exposes_no_images": True, "query_state_is_read_only": True}

    def close(self):
        for engine, _, _ in getattr(self, "banks", {}).values():
            engine.close()
