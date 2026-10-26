"""Real pinned MemoryOS layers and FAISS; all HTTP responses are mocked."""
import copy
import inspect
import json
from pathlib import Path
import shutil
from uuid import uuid4

import httpx
import pytest

pytest.importorskip("faiss")
from benchmarks import memoryos_adapter as adapter
from benchmarks import run_validation50_memoryos as runner
from benchmarks.comparison_worker import CachedClients, atomic_json
from benchmarks.run_budget import RunBudget
from memory_agent.clients import ModelError
from memory_agent.settings import Settings


@pytest.fixture
def workspace():
    root = Path(__file__).resolve().parents[1] / "runtime/memoryos_tests" / uuid4().hex
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root)


def observations(count=20):
    return [{"frame_index": n, "clock": f"06:00:{n:02d}", "description": f"Cup observation {n}",
        "entities": [{"id": "cup", "name": "cup", "kind": "object", "labels": [], "attributes": {}}],
        "relations": [], "interaction": None, "uncertainty": []} for n in range(count)]


def client_for(directory, sent, budget=None):
    source, _ = adapter.official_source()
    namespace = {}
    exec(compile((source / "prompts.py").read_text(encoding="utf-8"), str(source / "prompts.py"), "exec"), namespace)
    outputs = {
        namespace["CONTINUITY_CHECK_SYSTEM_PROMPT"]: "true",
        namespace["META_INFO_SYSTEM_PROMPT"]: "A cup was observed.",
        namespace["MULTI_SUMMARY_SYSTEM_PROMPT"]: json.dumps([
            {"theme": "Cup", "keywords": ["cup"], "content": "Cup observations"}]),
        namespace["PERSONALITY_ANALYSIS_SYSTEM_PROMPT"]: "None",
        namespace["KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT"]:
            "【User Private Data】\nNone\n【Assistant Knowledge】\nA cup was observed.",
    }
    def respond(request):
        payload = json.loads(request.content)
        sent.append(payload)
        assert "gold" not in payload and "gold_goal_frames" not in payload
        if "input" in payload:
            return httpx.Response(200, json={"data": [{"index": i, "embedding": [1., 1., .5]}
                for i, text in enumerate(payload["input"])], "usage": {"total_tokens": 2}})
        system = payload["messages"][0]["content"]
        if system in outputs:
            content = outputs[system]
        else:
            state = json.loads(payload["messages"][1]["content"][0]["text"])
            record = next(item for item in state["evidence"] if 11 in item["frame_indices"])
            content = json.dumps({"goal_frames": [11], "evidence_ids": [record["id"]]})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                       "usage": {"total_tokens": 10}})
    budget = budget or RunBudget(directory, max_requests=10000, max_reported_tokens=0)
    client = CachedClients(Settings(llm_key="fake", llm_model="fake", embedding_model="fake"),
        budget, directory / "response_cache", transport=httpx.MockTransport(respond))
    return client, budget


def method(root, client, frames=None):
    return adapter.MemoryOSMethod(root, root / "shared", {"ep": observations() if frames is None else frames},
                                  client, lambda **event: None)


def test_real_official_classes_and_source_tamper_rejection(workspace, monkeypatch):
    agent = adapter.OfficialMemoryOS(None, "ep", workspace)
    try:
        root, _ = adapter.official_source()
        assert agent.modules["updater"].Updater.process_short_term_to_mid_term.__code__.co_filename == str(root / "updater.py")
        assert agent.modules["utils"].gpt_generate_multi_summary.__code__.co_filename == str(root / "utils.py")
    finally:
        agent.close()
    monkeypatch.setattr(adapter, "SOURCE_HASHES", {"utils.py": "0" * 64})
    with pytest.raises(RuntimeError, match="官方文件被修改"):
        adapter.OfficialMemoryOS(None, "ep", workspace)


def test_all_layers_build_retrieve_readonly_and_resume_without_new_calls(workspace):
    sent = []
    client, budget = client_for(workspace, sent)
    bank = method(workspace, client)
    try:
        bank.build()
        path = workspace / "memoryos/ep/state.json"
        before = path.read_bytes()
        state = json.loads(before)
        assert state["processed"] == 20
        assert 0 < len(state["short_term"]) <= 10
        assert state["mid_term"]["sessions"]
        assert state["assistant_long_term"]["assistant_knowledge"]
        evidence, resolution, details = bank.retrieve("ep", "Find a cup.")
        assert resolution is None and details["query_state_is_read_only"]
        assert len(evidence) <= 20 and len(json.dumps(evidence).encode()) <= 40000
        assert any(item["content"]["layer"] == "long_term" for item in evidence)
        for item in evidence:
            if item["content"]["layer"].startswith("long_term"):
                assert item["frame_indices"] == []
            else:
                assert item["frame_indices"] == [item["content"]["original_observation"]["frame_index"]]
        bank.retrieve("ep", "Another cup question.")
        repeated, _, _ = bank.retrieve("ep", "Find a cup.")
        assert evidence == repeated and path.read_bytes() == before
        paid = len(sent)
        resumed = method(workspace, client)
        try:
            resumed.build()
            assert resumed.retrieve("ep", "Find a cup.")[0] == evidence
            assert len(sent) == paid == budget.state["requests_started"]
        finally:
            resumed.close()
    finally:
        bank.close()
        client.close()


def test_changed_observations_rejected_before_billing(workspace):
    sent = []
    client, _ = client_for(workspace, sent)
    bank = method(workspace, client, observations(3))
    bank.build()
    bank.close()
    changed = observations(3)
    changed[0]["description"] = "Changed source"
    bank = method(workspace, client, changed)
    try:
        with pytest.raises(RuntimeError, match="checkpoint does not match"):
            bank.build()
        assert not sent
    finally:
        bank.close()
        client.close()


def test_upstream_swallowed_failure_does_not_commit_partial_state(workspace, monkeypatch):
    sent = []
    client, budget = client_for(workspace, sent)
    post = client.post
    interrupted = [False]
    def interrupt(url, key, payload, role):
        response = post(url, key, payload, role)
        if role == "memoryos_topic_summary" and not interrupted[0]:
            interrupted[0] = True
            raise RuntimeError("Interrupted after paid response")
        return response
    monkeypatch.setattr(client, "post", interrupt)
    bank = method(workspace, client, observations(12))
    try:
        with pytest.raises((ModelError, RuntimeError)):
            bank.build()
        saved = json.loads((workspace / "memoryos/ep/state.json").read_text(encoding="utf-8"))
        assert saved["processed"] < 12
    finally:
        bank.close()
    paid = copy.deepcopy(sent)
    bank = method(workspace, client, observations(12))
    try:
        bank.build()
        assert json.loads((workspace / "memoryos/ep/state.json").read_text(encoding="utf-8"))["processed"] == 12
        keys = [json.dumps(payload, sort_keys=True) for payload in sent]
        assert len(keys) == len(set(keys)) == budget.state["requests_started"]
        assert sent[:len(paid)] == paid
    finally:
        bank.close()
        client.close()


def test_runner_common_answerer_and_complete_resume(workspace):
    shared, run = workspace / "shared", workspace / "experiment"
    atomic_json(shared / "perception/api_budget.json", {"requests_started": 0, "reported_tokens": 0})
    for n in range(12):
        path = shared / "frames/ep_1" / f"{n:03d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mock image")
    tasks = [{"row_idx": n, "ep_id": "ep_1", "task_id": f"task_{n}", "question": "Find a cup.",
              "answer": "[[11]]", "high_level_category": "Single-Goal Spatial Tasks",
              "low_level_category": "Object Recall"} for n in range(2)]
    sent = []
    def factory(settings, budget, cache):
        return client_for(run, sent, budget)[0]
    settings = Settings(llm_key="fake", llm_model="fake", embedding_model="fake")
    result = runner.evaluate(run, shared, tasks, {"ep_1": observations(12)}, settings, client_factory=factory)
    assert result["complete"] and result["correct"] == result["total"] == 2
    paid = len(sent)
    repeated = runner.evaluate(run, shared, tasks, {"ep_1": observations(12)}, settings, client_factory=factory)
    assert repeated["api_usage"]["requests_started"] == paid == len(sent)
    assert (run / "REPORT.md").read_text(encoding="utf-8").startswith("# MemoryOS")
    assert list(inspect.signature(adapter.MemoryOSMethod.retrieve).parameters) == ["self", "episode", "question"]
