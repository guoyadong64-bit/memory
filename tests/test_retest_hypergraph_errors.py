import hashlib
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import httpx
import pytest

from benchmarks import retest_hypergraph_errors as runner
from benchmarks.comparison_worker import atomic_json
from benchmarks.run_budget import RunBudget
from memory_agent.settings import Settings


@pytest.fixture(autouse=True, params=["v2", "v3", "v4"])
def runner_variant(request, monkeypatch):
    if request.param == "v3":
        from benchmarks import retest_hypergraph_errors_v3
        monkeypatch.setattr(sys.modules[__name__], "runner", retest_hypergraph_errors_v3)
    elif request.param == "v4":
        from benchmarks import retest_hypergraph_errors_v4
        monkeypatch.setattr(sys.modules[__name__], "runner", retest_hypergraph_errors_v4)


@pytest.fixture
def workspace():
    root = Path(__file__).resolve().parents[1] / "runtime/retest_runner_tests" / uuid4().hex
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root)


def questions():
    return [{"row_idx": n, "ep_id": "ep_1", "task_id": f"task_{n}", "question": f"question {n}",
             "answer": json.dumps([[n]]), "high_level_category": "Spatial", "low_level_category": "Recall"}
            for n in range(3)]


def test_baseline_selects_only_verified_errors_and_rejects_model_change(workspace):
    source, shared = workspace / "source", workspace / "shared"
    tasks = questions()
    settings = Settings(llm_model="fake")
    atomic_json(shared / "observations_manifest.json", {"sealed": True})
    names = ("findingdory_visual_answer.py", "answer_normalization.py", "findingdory_pilot.py",
             "findingdory_mem0.py", "findingdory_memory.py", "run_validation50_hypergraph.py")
    atomic_json(source / "config.json", {"method": "hypergraph", "tasks_sha256": runner.fingerprint(tasks),
        "observations_manifest_sha256": runner.common.checksum(shared / "observations_manifest.json"),
        "models": {"llm_model": "fake"}, "source_sha256": {
            "benchmarks/" + name: runner.common.checksum(runner.REPO / "benchmarks" / name) for name in names}})
    atomic_json(source / "summary.json", {"complete": True, "correct": 1})
    for task in tasks:
        atomic_json(source / "answers" / f"{task['row_idx']}.json", {
            "method": "hypergraph", "row_idx": task["row_idx"], "task_sha256": runner.fingerprint(task),
            "prediction": {"goal_frames": [0]}, "evidence": [{"id": "fact", "frame_indices": [0]}],
            "correct": task["row_idx"] == 0, "candidate_frame_covered": float(task["row_idx"] == 0)})
    selected, _ = runner.baseline_errors(source, tasks, shared, settings)
    assert [t["row_idx"] for t in selected] == [1, 2]
    with pytest.raises(RuntimeError, match="不一致"):
        runner.baseline_errors(source, tasks, shared, settings.model_copy(update={"llm_model": "changed"}))


def test_old_embedding_cache_is_reused_but_old_answer_is_not(workspace):
    source = workspace / "source_cache"
    sent = []
    budget = RunBudget(workspace / "run", max_requests=100, max_reported_tokens=0)
    client = runner.RetestClients(Settings(llm_key="fake"), budget, workspace / "new_cache", source_cache=source,
        transport=httpx.MockTransport(lambda request: sent.append(request) or httpx.Response(200,
            json={"model": "fake", "usage": {"total_tokens": 2}})))
    for role in ["embedding", "answer"]:
        payload = {"model": "fake", "input": "test"}
        digest = hashlib.sha256(json.dumps({"url": "https://mock/api", "role": role, "payload": payload},
            sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        atomic_json(source / f"{digest}.json", {"model": "fake", "source_marker": "old"})
        response = client.post("https://mock/api", "fake", payload, role)
        assert (response.get("source_marker") == "old") == (role == "embedding")
    assert len(sent) == budget.state["requests_started"] == 1
    client.close()


def test_error_only_resume_caches_completed_requests_and_never_claims_overall_accuracy(workspace):
    tasks = questions()[1:]
    baseline = {n: {"correct": n == 0, "candidate_frame_covered": float(n == 0)} for n in range(3)}
    sent = []
    crash = [True]

    def transport(request):
        payload = json.loads(request.content)
        assert "gold_goal_frames" not in payload and "answer" not in payload
        sent.append(payload)
        return httpx.Response(200, json={"model": "fake", "usage": {"total_tokens": 10}})

    def factory(settings, budget, cache, **kwargs):
        return runner.RetestClients(settings, budget, cache, transport=httpx.MockTransport(transport), **kwargs)

    class Method:
        def __init__(self, directory, shared, observations, client, progress):
            self.client = client
            assert observations == {"ep_1": []}

        def build(self):
            pass

        def retrieve(self, ep, question):
            self.client.post("https://mock/api", "fake", {"model": "fake", "input": question}, "embedding")
            return [{"id": "fact", "frame_indices": [1]}], {}, {}

        def close(self):
            pass

    def decoder(client, question, *args):
        client.post("https://mock/api", "fake", {"model": "fake", "question": question}, "answer")
        if question == "question 2" and crash[0]:
            crash[0] = False
            raise RuntimeError("simulated interruption")
        return {"goal_frames": [1]}

    kwargs = dict(method_factory=Method, client_factory=factory, decoder=decoder)
    args = (workspace / "run", workspace / "source", workspace / "shared", tasks,
            {"ep_1": []}, Settings(llm_key="fake"), baseline)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.evaluate(*args, **kwargs)
    assert runner.load(workspace / "run/summary.json")["completed"] == 1
    result = runner.evaluate(*args, **kwargs)
    assert result["complete"] and result["correct"] == 1 and result["total"] == 2
    assert result["new_overall_accuracy"] is None
    assert len(sent) == result["api_usage"]["requests_started"] == 4
    runner.evaluate(*args, **kwargs)
    assert len(sent) == 4
