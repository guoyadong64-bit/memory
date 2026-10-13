"""Official Mem0 OSS algorithm with budgeted provider transports.

Only LLM/embedding HTTP transport is adapted; official add(infer=True),
fact extraction, dedup, entity linking and search remain in the pinned SDK.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from benchmarks.run_budget import BudgetExceeded

MEM0_VERSION = "2.2.1"
_client = None


class SDKBudgetStop(BaseException):
    """Must cross SDK broad Exception fallbacks without hidden additional calls."""


class BudgetedMem0LLM:
    def __init__(self, config):
        self.config = config

    def generate_response(self, messages, tools=None, tool_choice="auto", **kwargs):
        if tools:
            raise ValueError("This extraction adapter does not implement tool calls")
        settings = _client.settings
        payload = {"model":settings.llm_model,"messages":messages,"temperature":0,
                   "max_tokens":4096,"response_format":kwargs.get("response_format", {"type":"json_object"}),
                   **_client.chat_options(settings.llm_model)}
        try:
            result = _client.post(settings.llm_url.rstrip("/")+"/chat/completions",
                                  settings.llm_key,payload,"mem0_extract")
        except BudgetExceeded as exc:
            raise SDKBudgetStop(str(exc)) from None
        return result["choices"][0]["message"]["content"]


class BudgetedMem0Embedding:
    def __init__(self, config):
        self.config = config

    def embed(self, text, memory_action="add"):
        return self.embed_batch([text], memory_action)[0]

    def embed_batch(self, texts, memory_action="add"):
        try:
            return _client.embed(list(texts))
        except BudgetExceeded as exc:
            raise SDKBudgetStop(str(exc)) from None


def observation_message(frame):
    # All input comes from shared perception; no question or answer annotations.
    return {"role":"user", "content":"Historical robot observation: " +
        json.dumps({k:frame[k] for k in ("frame_index","clock","description","entities",
                                         "relations","interaction","uncertainty")}, ensure_ascii=False)}


def memory_frame_indices(item):
    """A batch is provenance, not proof that each extracted fact covers every frame."""
    declared=set(item.get("metadata",{}).get("frame_indices",[]))
    mentioned=set()
    for match in re.finditer(r"\bframes?(?:_index|\s+index)?\s*[=:]?\s*(\d+)"
                            r"(?:\s*(?:-|–|to)\s*(\d+))?",item["memory"],re.I):
        first=int(match.group(1));last=int(match.group(2)) if match.group(2) else first
        if 0<=first<=last<96:mentioned.update(range(first,last+1))
    return sorted(mentioned & declared)


def create_memory(path: Path, client, dimensions: int):
    global _client
    _client = client
    path.mkdir(parents=True,exist_ok=True)
    os.environ["MEM0_TELEMETRY"] = "false"
    os.environ["MEM0_DIR"] = str((path/"sdk_config").resolve())
    from importlib.metadata import version
    if version("mem0ai") != MEM0_VERSION:
        raise RuntimeError("Install the pinned Mem0 SDK from requirements-benchmark.txt")
    from mem0 import Memory
    from mem0.utils.factory import LlmFactory, EmbedderFactory
    # Supported provider hooks change transport, never the memory algorithm.
    LlmFactory.provider_to_class["openai"] = (
        "benchmarks.mem0_adapter.BudgetedMem0LLM",LlmFactory.provider_to_class["openai"][1])
    EmbedderFactory.provider_to_class["openai"] = "benchmarks.mem0_adapter.BudgetedMem0Embedding"
    return Memory.from_config({
        "llm":{"provider":"openai","config":{"model":client.settings.llm_model,"temperature":0}},
        "embedder":{"provider":"openai","config":{"model":client.settings.embedding_model,
                                                      "embedding_dims":dimensions}},
        "vector_store":{"provider":"qdrant","config":{"path":str((path/"qdrant").resolve()),
                          "collection_name":"robot_history","embedding_model_dims":dimensions}},
        "history_db_path":str((path/"history.sqlite3").resolve()),
        "custom_instructions":(
            "Remember the supplied historical robot observations, not personal preferences. "
            "Preserve episode identity, sampled frame_index, visible clock, entity IDs and "
            "category labels, object support relations, actions, sources and destinations. "
            "Keep earlier locations as historical facts rather than overwriting them. "
            "Distinguish the manipulated object from background mentions. Keep uncertainty. "
            "Every fact should retain its exact supporting frame_index or frame indices. "
            "Never infer unseen actions, aliases, places, or navigation distances.")})
