"""Validate resumption, sealed inputs and scoring without live API requests."""
import json
from pathlib import Path
import shutil
from uuid import uuid4

import httpx
import pytest

from benchmarks import run_validation50_hypergraph as runner
from benchmarks.comparison_worker import CachedClients, atomic_json
from memory_agent.settings import Settings


@pytest.fixture
def workspace_dir():
    root = Path(__file__).resolve().parents[1] / "runtime/validation50_runner_tests" / uuid4().hex
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root)


def tasks():
    return [{"row_idx": i, "ep_id": "ep_1" if i < 2 else "ep_2", "task_id": f"task_{i}",
             "question": f"question {i}", "video": f"videos/val/ep_{1 if i < 2 else 2}.mp4",
             "answer": json.dumps([[1 if i == 1 else 0]]),
             "high_level_category": "Single-Goal Spatial Tasks", "low_level_category": "Object Recall"}
            for i in range(3)]


def shared_fixture(root):
    shared = root / "shared"
    atomic_json(shared / "perception/api_budget.json", {"requests_started": 5, "reported_tokens": 50})
    return shared


@pytest.mark.parametrize("method_name", ["hypergraph", "mem0", "readagent"])
def test_failure_resume_reuses_paid_responses_and_scores_consistently(workspace_dir, capsys, monkeypatch, method_name):
    if method_name == "mem0":
        from benchmarks import run_validation50_mem0 as tested
    elif method_name == "readagent":
        from benchmarks import run_validation50_readagent as tested
        monkeypatch.setattr(tested, "official_source", lambda: ({}, {"revision": "fake-official-revision"}))
    else:
        tested = runner
    run = workspace_dir / "run"
    shared = shared_fixture(workspace_dir)
    sent = []

    def transport(request):
        payload = json.loads(request.content)
        sent.append(payload)
        assert "answer" not in payload and "gold_goal_frames" not in payload
        return httpx.Response(200, json={"model": "fake", "usage": {"total_tokens": 10}})

    def client_factory(settings, budget, directory):
        return CachedClients(settings, budget, directory, transport=httpx.MockTransport(transport))

    class Method:
        def __init__(self, directory, data, observations, client, progress):
            self.client = client
            self.progress = progress
            # The memory interface gets visual observations, not task/answer labels.
            assert all(set(f) == {"frame_index", "description"}
                       for frames in observations.values() for f in frames)

        def build(self):
            self.progress(completed=1, total=1)

        def retrieve(self, ep, question):
            self.client.post("https://mock/embedding", "fake", {"model": "fake", "input": ep + question}, "embedding")
            return [{"id": "fact", "content": "observation", "frame_indices": [0]}], None, {}

        def close(self):
            pass

    crash = [True]

    def decoder(client, question, *args):
        client.post("https://mock/answer", "fake", {"model": "fake", "question": question}, "answer")
        if question == "question 1" and crash[0]:
            crash[0] = False
            raise RuntimeError("Interrupted after paid response, before question checkpoint")
        return {"goal_frames": [0], "evidence_ids": ["fact"]}

    observations = {ep: [{"frame_index": 0, "description": "a chair"}] for ep in ("ep_1", "ep_2")}
    settings = Settings(llm_key="fake", embedding_model="fake", embedding_key="fake")
    kwargs = dict(method_factory=Method, client_factory=client_factory, decoder=decoder)
    with pytest.raises(RuntimeError, match="Interrupted"):
        tested.evaluate(run, shared, tasks(), observations, settings, **kwargs)
    partial = runner.load(run / "summary.json")
    assert partial["completed"] == 1 and not partial["complete"]
    assert len(sent) == 4
    result = tested.evaluate(run, shared, tasks(), observations, settings, **kwargs)
    assert result["completed"] == 3 and result["correct"] == 2
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["api_usage"]["requests_started"] == 6 == len(sent)
    assert result["api_usage"]["reported_tokens"] == 60
    assert len((run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert {json.loads(line)["method"] for line in
            (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()} == {method_name}
    assert "2/3" in capsys.readouterr().out
    assert "66.67%" in (run / "REPORT.md").read_text(encoding="utf-8")
    tested.evaluate(run, shared, tasks(), observations, settings, **kwargs)
    assert len(sent) == 6  # A fully completed invocation makes no HTTP calls.
    checkpoint = run / "answers/0.json"
    row = runner.load(checkpoint)
    row["correct"] = False
    atomic_json(checkpoint, row)
    with pytest.raises(RuntimeError, match="评分被改动"):
        runner.load_answers(run, tasks())


def test_sealed_source_and_images_are_verified(workspace_dir, monkeypatch):
    shared = shared_fixture(workspace_dir)
    data = workspace_dir / "data"
    data.mkdir()
    manifest = {"episodes": ["ep_1"], "revision": "fixed", "videos": {
        "videos/val/ep_1.mp4": {"sha256": "video"}}}
    monkeypatch.setattr(runner, "selected_data", lambda directory: manifest)
    ep = shared / "frames/ep_1"
    ep.mkdir(parents=True)
    for i in range(96):
        (ep / f"{i:03d}.jpg").write_bytes(f"image {i}".encode())
    atomic_json(ep / "frames_manifest.json", {
        "source_video": "videos/val/ep_1.mp4", "video_sha256": "video",
        "frame_sha256": {f"{i:03d}.jpg": runner.checksum(ep / f"{i:03d}.jpg") for i in range(96)}})
    atomic_json(shared / "config.json", {"fingerprint": "frozen", "dataset": manifest})
    observation = shared / "observations/ep_1.json"
    atomic_json(observation, {"episode": "ep_1", "schema_version": runner.SCHEMA_VERSION,
        "configuration_fingerprint": "frozen", "frame_manifest_sha256": runner.checksum(ep / "frames_manifest.json"),
        "observations": [{"frame_index": i, "description": "a chair"} for i in range(96)]})
    atomic_json(shared / "observations_manifest.json", {
        "protocol": "findingdory-validation50-shared-perception-v1", "episodes": ["ep_1"],
        "dataset_revision": "fixed", "configuration_fingerprint": "frozen",
        "observation_sha256": {"ep_1": runner.checksum(observation)}})
    (data / "validation_first50_questions.jsonl").write_text(
        json.dumps(tasks()[0]) + "\n", encoding="utf-8")
    _, questions, observations = runner.verify_inputs(data, shared)
    assert len(questions) == 1 and len(observations["ep_1"]) == 96
    (ep / "000.jpg").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="原图被改动"):
        runner.verify_inputs(data, shared)


def test_checkpoint_rejects_changed_questions(workspace_dir):
    run = workspace_dir / "run"
    atomic_json(run / "answers/0.json", {"row_idx": 0, "task_sha256": "different"})
    with pytest.raises(RuntimeError, match="断点与冻结题目不一致"):
        runner.load_answers(run, tasks())


@pytest.mark.parametrize("gold", [[[-1]], [[]]])
def test_missing_official_goal_frames_are_not_dropped_or_matched_as_sentinels(gold):
    task = tasks()[0] | {"answer": json.dumps(gold)}
    assert not runner.has_valid_targets(task)
    assert runner.score_result(task, {"goal_frames": [-1]}, [
        {"id": "fact", "frame_indices": [0]}]) == (False, 0)
    assert runner.score_result(task, {"goal_frames": []}, []) == (False, 0)


def test_one_run_directory_cannot_be_opened_twice(workspace_dir):
    with runner.run_lock(workspace_dir):
        with pytest.raises(RuntimeError, match="已有进程运行"):
            with runner.run_lock(workspace_dir):
                pass


def test_mem0_reference_rejects_model_change_before_requests(workspace_dir, monkeypatch):
    from benchmarks import run_validation50_mem0 as mem0
    shared = workspace_dir / "shared"
    atomic_json(shared / "observations_manifest.json", {"sealed": True})
    settings = Settings(llm_model="original")
    atomic_json(workspace_dir / "reference/config.json", {
        "models": {"llm_model": "different"}, "tasks_sha256": runner.fingerprint(tasks()),
        "observations_manifest_sha256": runner.checksum(shared / "observations_manifest.json")})
    monkeypatch.setattr(runner, "DEFAULT_RUN", workspace_dir / "reference")
    with pytest.raises(RuntimeError, match="不一致"):
        mem0.verify_reference(shared, settings, tasks())
