"""Offline protocol tests: no provider or dataset-network calls."""
import json
from pathlib import Path

import httpx
import pytest

from benchmarks import comparison_worker as worker
from benchmarks import run_comparison as runner
from benchmarks import run_budget
from benchmarks.comparison_methods import MemoryMethod, REGISTRY, register
from benchmarks.comparison_worker import atomic_json
from benchmarks.run_budget import BudgetedClients, RunBudget
from memory_agent.clients import ModelClients, ModelError
from memory_agent.settings import Settings


def fixture_run(tmp_path, monkeypatch):
    run, data = tmp_path / "run", tmp_path / "data"
    tasks = [{"row_idx": n, "task_id": f"t{n}", "ep_id": "ep_1" if n < 6 else "ep_33",
              "question": str(n), "answer": json.dumps([[n]]),
              "low_level_category": "Single"} for n in range(12)]
    atomic_json(data / "manifest.json", {"episodes": ["ep_1", "ep_33"]})
    atomic_json(data / "selected_tasks.json", tasks)
    for ep in ("ep_1", "ep_33"):
        atomic_json(run / "observations" / f"{ep}.json", {"observations": []})
    monkeypatch.setattr(worker.Settings, "load", classmethod(
        lambda cls: Settings(llm_key="fake", embedding_key="fake", embedding_model="fake")))
    calls = {"answers": 0, "fail_at": None}

    def answer(client, question, evidence, observations, data_dir, ep, resolution):
        calls["answers"] += 1
        if calls["fail_at"] == int(question):
            raise RuntimeError("injected failure")
        client.budget.begin()
        client.budget.finish("answer", "fake", {"total_tokens": 10})
        return {"goal_frames": [int(question)], "visual_frames": [int(question)],
                "evidence_ids": [evidence[0]["id"]]}

    monkeypatch.setattr(worker, "decode_visual", answer)

    class Fake(MemoryMethod):
        name = "fake"

        def build(self):
            self.progress(stage="memory", episode="ep_1", completed=1, total=1)

        def retrieve(self, episode, question):
            return [{"id": f"e{question}", "content": "fake",
                     "frame_indices": [int(question)]}], None, {"mock": True}

    monkeypatch.setitem(REGISTRY, "fake", Fake)
    return run, data, tasks, calls


def test_12_answers_resume_and_report(tmp_path, monkeypatch, capsys):
    run, data, tasks, calls = fixture_run(tmp_path, monkeypatch)
    for name in ("hypergraph", "mem0", "amem", "readagent"):
        monkeypatch.setitem(REGISTRY, name, REGISTRY["fake"])
        worker.run_method(run, data, name)
    assert calls["answers"] == 48
    for name in ("hypergraph", "mem0", "amem", "readagent"):
        worker.run_method(run, data, name)
    assert calls["answers"] == 48
    RunBudget(run / "perception", max_requests=5000, max_reported_tokens=0)
    summaries = runner.summarize(run, tasks, ("hypergraph", "mem0", "amem", "readagent"))
    assert all(item["correct"] == 12 and item["reported_tokens"] == 120 for item in summaries)
    assert len((run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 48
    report = (run / "REPORT.md").read_text(encoding="utf-8")
    assert report.count("12/12") == 4
    assert capsys.readouterr().out.count("12/12") == 4


def test_failure_checkpoint_resume_and_plugin_registry(tmp_path, monkeypatch):
    run, data, tasks, calls = fixture_run(tmp_path, monkeypatch)
    calls["fail_at"] = 6
    assert worker.main(["--run-dir", str(run), "--data-dir", str(data), "--method", "fake"]) == 1
    assert (run / "methods" / "fake" / "last_failure.json").exists()
    assert len(list((run / "methods" / "fake" / "answers").glob("*.json"))) == 6
    calls["fail_at"] = None
    assert worker.main(["--run-dir", str(run), "--data-dir", str(data), "--method", "fake"]) == 0
    assert calls["answers"] == 13  # six completed questions were not requested again
    assert len(list((run / "methods" / "fake" / "answers").glob("*.json"))) == 12
    assert REGISTRY["fake"].name == "fake"
    class Added(REGISTRY["fake"]):
        name = "added_method_for_test"

    try:
        register(Added)
        worker.run_method(run, data, Added.name)
        assert len(list((run / "methods" / Added.name / "answers").glob("*.json"))) == 12
    finally:
        REGISTRY.pop(Added.name, None)


def test_response_cache_avoids_second_model_request(tmp_path, monkeypatch):
    budget = RunBudget(tmp_path / "budget", max_requests=5000, max_reported_tokens=0)
    calls = []

    def fake_post(self, url, key, payload, role):
        calls.append(role)
        budget.begin()
        budget.finish(role, payload["model"], {"total_tokens": 9})
        return {"model": payload["model"], "usage": {"total_tokens": 9}, "choices": []}

    monkeypatch.setattr(BudgetedClients, "post", fake_post)
    client = worker.CachedClients(Settings(llm_key="fake"), budget, tmp_path / "cache")
    try:
        first = client.post("https://example.test", "fake", {"model": "fake", "x": 1}, "answer")
        second = client.post("https://example.test", "fake", {"model": "fake", "x": 1}, "answer")
    finally:
        client.close()
    assert first == second
    assert calls == ["answer"]
    assert budget.state["requests_started"] == 1


def test_live_no_retry_client_waits_for_slow_response_without_replaying():
    attempts = []

    def fail_once(request):
        attempts.append(request)
        raise httpx.ReadTimeout("simulated slow response")

    client = ModelClients(Settings(mode="live", timeout=60, retries=0),
                          httpx.MockTransport(fail_once))
    try:
        assert client.http.timeout.read == 180
        with pytest.raises(ModelError, match="ReadTimeout"):
            client.post("https://example.test/chat/completions", "fake", {"model": "fake"}, "visual_observation")
        assert len(attempts) == 1
    finally:
        client.close()


def test_remote_disconnect_retries_once_and_counts_each_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(run_budget.time, "sleep", lambda _: None)
    attempts = []

    def respond(request):
        attempts.append(request)
        if len(attempts) == 1:
            raise httpx.RemoteProtocolError("remote closed")
        return httpx.Response(200, json={"model": "fake", "usage": {"total_tokens": 11}})

    budget = RunBudget(tmp_path / "budget", max_requests=5000, max_reported_tokens=0)
    client = BudgetedClients(Settings(mode="live", timeout=60, retries=0), budget,
                             transport=httpx.MockTransport(respond))
    try:
        result = client.post("https://example.test", "fake", {"model": "fake"}, "visual_observation")
    finally:
        client.close()
    assert result["usage"]["total_tokens"] == 11
    assert len(attempts) == 2
    assert budget.state["requests_started"] == 2
    assert budget.state["failed_requests"] == 1
    assert budget.state["reported_tokens"] == 11
    assert "RemoteProtocolError" in budget.log.read_text(encoding="utf-8")


def test_repeated_disconnect_stops_after_two_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(run_budget.time, "sleep", lambda _: None)
    attempts = []

    def respond(request):
        attempts.append(request)
        raise httpx.RemoteProtocolError("remote closed")

    budget = RunBudget(tmp_path / "budget", max_requests=5000, max_reported_tokens=0)
    client = BudgetedClients(Settings(mode="live", timeout=60, retries=0), budget,
                             transport=httpx.MockTransport(respond))
    try:
        with pytest.raises(ModelError, match="RemoteProtocolError"):
            client.post("https://example.test", "fake", {"model": "fake"}, "visual_observation")
    finally:
        client.close()
    assert len(attempts) == 2
    assert budget.state["requests_started"] == 2


def test_budget_rename_recovers_from_transient_windows_reader_lock(tmp_path, monkeypatch):
    budget = RunBudget(tmp_path / "budget", max_requests=5000, max_reported_tokens=0)
    original_replace = Path.replace
    attempts = []
    monkeypatch.setattr(run_budget.time, "sleep", lambda _: None)

    def locked_once(path, target):
        if path == budget.path.with_suffix(".tmp"):
            attempts.append(1)
            if len(attempts) == 1:
                raise PermissionError(5, "temporary Windows sharing violation")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", locked_once)
    budget.begin()
    assert len(attempts) == 2
    assert json.loads(budget.path.read_text(encoding="utf-8"))["requests_started"] == 1


def test_remote_disconnect_then_connect_error_can_recover(tmp_path, monkeypatch):
    monkeypatch.setattr(run_budget.time, "sleep", lambda _: None)
    attempts = []

    def respond(request):
        attempts.append(request)
        if len(attempts) == 1:
            raise httpx.RemoteProtocolError("remote closed")
        if len(attempts) == 2:
            raise httpx.ConnectError("cannot connect")
        return httpx.Response(200, json={"model": "fake", "usage": {"total_tokens": 7}})

    budget = RunBudget(tmp_path / "budget", max_requests=5000, max_reported_tokens=0)
    client = BudgetedClients(Settings(mode="live", retries=0), budget,
                             transport=httpx.MockTransport(respond))
    try:
        result = client.post("https://example.test", "fake", {"model": "fake"}, "answer")
    finally:
        client.close()
    assert result["usage"]["total_tokens"] == 7
    assert len(attempts) == 3
    assert budget.state["requests_started"] == 3
    assert budget.state["failed_requests"] == 2


def test_parent_progress_does_not_read_active_budget(tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    perception = RunBudget(run / "perception", max_requests=5000, max_reported_tokens=0)
    earlier = RunBudget(run / "methods" / "hypergraph", max_requests=5000, max_reported_tokens=0)
    RunBudget(run / "methods" / "mem0", max_requests=5000, max_reported_tokens=0)
    perception.begin()
    perception.finish("vision", "fake", {"total_tokens": 11})
    earlier.begin()
    earlier.finish("answer", "fake", {"total_tokens": 13})
    original_load = runner.load

    def guarded_load(path):
        if Path(path).parent.name == "mem0":
            raise AssertionError("parent tried to read the active worker budget")
        return original_load(path)

    monkeypatch.setattr(runner, "load", guarded_load)

    class FakeProcess:
        stdout = iter(['@@PROGRESS@@{"stage":"answer","episode":"ep_1",'
                       '"completed":4,"total":12,"correct":2,"requests":3,"tokens":17}\n'])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self):
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    runner.execute_method(run, tmp_path, "mem0", ())
    output = capsys.readouterr().out
    assert "总请求 5" in output
    assert "累计 token 41" in output
