"""Mock-only recovery tests; no real model requests or historical run writes."""
from copy import deepcopy
import json

import httpx
import pytest

from benchmarks import resume_hypergraph_remaining_v5 as recovery
from benchmarks.comparison_worker import atomic_json
from benchmarks.findingdory_structured import fingerprint
from benchmarks.run_budget import RunBudget
from memory_agent.clients import ModelError
from memory_agent.settings import Settings


def settings():
    return Settings(mode="live", llm_key="test-key", llm_url="https://offline.test/v1",
                    llm_model="mock", retries=0)


def request():
    return {"model": "mock", "temperature": 0, "max_tokens": 3072,
            "response_format": {"type": "json_object"}, "messages": [
                {"role": "system", "content": "Same answer rules"},
                {"role": "user", "content": [
                    {"type": "text", "text": "Same question and evidence"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA", "detail": "high"}}]}]}


def response(content):
    return {"model": "mock", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "choices": [{"finish_reason": "stop", "message": {"content": content}}]}


def valid(frame=0):
    return response(json.dumps({"goal_frames": [frame], "evidence_ids": ["fact"],
                               "target_bindings": [{"frame_index": frame, "entity_id": "cup", "evidence_ids": ["fact"]}]}))


def make_client(run, handler):
    return recovery.RecoveryClients(settings(), RunBudget(run, max_requests=1_000_000, max_reported_tokens=0),
                                   run / "response_cache", transport=httpx.MockTransport(handler))


def cached(run, payload, result, role="answer"):
    key = recovery.request_hash("https://offline.test/v1/chat/completions", payload, role)
    path = run / "response_cache" / f"{key}.json"
    atomic_json(path, result)
    return path


@pytest.mark.parametrize("content", ['{"goal_frames":[85,84,67,', '```json\n{}\n```', '[]', 'null', ''])
def test_invalid_json_cache_is_preserved_and_success_replays_without_billing(tmp_path, content):
    sent, payload = [], request()
    before_payload = deepcopy(payload)
    path = cached(tmp_path, payload, response(content))
    original_bytes = path.read_bytes()
    def handler(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json=valid())
    client = make_client(tmp_path, handler)
    try:
        result = client.post("https://offline.test/v1/chat/completions", "test-key", payload, "answer")
        assert recovery.format_error(result) is None
        assert len(sent) == 1 and client.budget.state["requests_started"] == 1
        assert payload == before_payload
        repaired = sent[0]
        assert {k: v for k, v in repaired.items() if k != "messages"} == {k: v for k, v in payload.items() if k != "messages"}
        assert repaired["messages"][0] == payload["messages"][0]
        assert repaired["messages"][1]["content"][:-1] == payload["messages"][1]["content"]
    finally:
        client.close()
    resumed = make_client(tmp_path, handler)
    try:
        assert resumed.post("https://offline.test/v1/chat/completions", "test-key", payload, "answer") == result
        assert len(sent) == 1 and resumed.budget.state["reported_tokens"] == 15
    finally:
        resumed.close()
    assert path.read_bytes() == original_bytes


def test_valid_and_non_answer_responses_are_unchanged(tmp_path):
    payload = request()
    cached(tmp_path, payload, valid())
    cached(tmp_path, payload, {"data": [{"embedding": [1, 0], "index": 0}]}, role="embedding")
    def forbidden(req): raise AssertionError("No HTTP request expected")
    client = make_client(tmp_path, forbidden)
    try:
        assert client.post("https://offline.test/v1/chat/completions", "test-key", payload, "answer") == valid()
        assert "data" in client.post("https://offline.test/v1/chat/completions", "test-key", payload, "embedding")
        assert client.budget.state["requests_started"] == 0
        assert not client.recovery_dir.exists()
    finally:
        client.close()


def test_two_attempts_stop_and_manual_restart_gets_fresh_round(tmp_path):
    payload, sent = request(), []
    cached(tmp_path, payload, response('{"goal_frames":['))
    def invalid(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json=response('{"goal_frames":[0,'))
    client = make_client(tmp_path, invalid)
    try:
        with pytest.raises(ModelError, match="two format retries"):
            client.post("https://offline.test/v1/chat/completions", "test-key", payload, "answer")
        assert len(sent) == 2
    finally:
        client.close()
    def accepted(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json=valid())
    resumed = make_client(tmp_path, accepted)
    try:
        assert recovery.format_error(resumed.post("https://offline.test/v1/chat/completions", "test-key", payload, "answer")) is None
        assert len(sent) == 3
        assert resumed.budget.state["reported_tokens"] == 45
        instruction = json.loads(sent[-1]["messages"][-1]["content"][-1]["text"])
        assert instruction["format_recovery"]["round"] == 1
    finally:
        resumed.close()


def test_network_timeout_is_not_a_format_retry(tmp_path):
    calls = []
    def timeout(req):
        calls.append(req)
        raise httpx.ReadTimeout("simulated", request=req)
    client = make_client(tmp_path, timeout)
    try:
        with pytest.raises(ModelError, match="ReadTimeout"):
            client.post("https://offline.test/v1/chat/completions", "test-key", request(), "answer")
        assert len(calls) == 1 and not client.recovery_dir.exists()
    finally:
        client.close()


def test_complete_checkpoint_preserved_and_crash_after_recovery_does_not_rebill(tmp_path, monkeypatch):
    run, shared = tmp_path / "run", tmp_path / "shared"
    tasks = [{"row_idx": n, "ep_id": "ep_1", "task_id": f"task{n}", "question": "Navigate to a cup.",
              "answer": json.dumps([[n]]), "high_level_category": "Spatial", "low_level_category": "Recall"} for n in range(2)]
    atomic_json(run / "answers/0.json", {"row_idx": 0, "method": "hypergraph_recall_v5",
        "task_sha256": fingerprint(tasks[0]), "prediction": {"goal_frames": [0]},
        "evidence": [{"id": "old", "frame_indices": [0]}], "correct": True, "candidate_frame_covered": 1.0})
    before = (run / "answers/0.json").read_bytes()
    frame = shared / "frames/ep_1/001.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"mock-image-not-interpreted")
    sent = []
    def handler(req):
        payload = json.loads(req.content)
        sent.append(payload)
        return httpx.Response(200, json=response('{"goal_frames":[1,') if len(sent) == 1 else valid(1))
    def factory(settings, budget, cache, **kwargs):
        return recovery.RecoveryClients(settings, budget, cache, transport=httpx.MockTransport(handler), **kwargs)
    class Method:
        def __init__(self, *args): pass
        def build(self): pass
        def close(self): pass
        def retrieve(self, ep, question):
            return [{"id": "fact", "frame_indices": [1], "content": {"query_frame_priority": [1]}}], None, {"read_projection_v5": {}}
    interrupted = [False]
    original_write = recovery.original.atomic_json
    def write(path, value):
        if path.name == "1.json" and not interrupted[0]:
            interrupted[0] = True
            raise RuntimeError("crash after accepted response")
        return original_write(path, value)
    monkeypatch.setattr(recovery.original, "atomic_json", write)
    params = (run, tmp_path / "source", shared, tasks,
              {"ep_1": [{"frame_index": 1, "entities": [{"id": "cup"}]}]}, settings(),
              {0: {"correct": False}, 1: {"correct": False}})
    kwargs = {"method_factory": Method, "client_factory": factory}
    with pytest.raises(RuntimeError, match="crash after accepted"):
        recovery.original.evaluate(*params, **kwargs)
    assert len(sent) == 2
    result = recovery.original.evaluate(*params, **kwargs)
    assert result["complete"] and result["correct"] == 2
    assert len(sent) == 2 and (run / "answers/0.json").read_bytes() == before
    assert result["new_overall_accuracy"] is None
    recovery.annotate_report(run)
    summary = recovery.original.common.load(run / "summary.json")
    assert summary["answer_format_recovery"]["recovered_requests"] == 1
    assert summary["api_usage"]["requests_started"] == 2
    assert "JSON 输出恢复记录" in (run / "REPORT.md").read_text(encoding="utf-8")


def test_structurally_wrong_answer_still_uses_original_validation(tmp_path):
    from benchmarks.hypergraph_visual_answer_v5 import decode_visual
    frame = tmp_path / "frames/ep_1/000.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"mock")
    sent = []
    def invalid_target(req):
        sent.append(req)
        return httpx.Response(200, json=valid(95))
    client = make_client(tmp_path / "run", invalid_target)
    try:
        answer = decode_visual(client, "Navigate to a cup.", [{"id": "fact", "frame_indices": [0], "content": {}}],
                               [{"frame_index": 0, "entities": [{"id": "cup"}]}], tmp_path, "ep_1")
        assert answer["goal_frames"] == [] and "validation" in answer
        assert len(sent) == 2  # exactly the pre-existing semantic-validation loop
        assert not client.recovery_dir.exists()
    finally:
        client.close()
