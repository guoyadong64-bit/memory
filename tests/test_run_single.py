"""Standalone runner tests never call a model API or download the dataset."""
import json
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from benchmarks import run_amem, run_hypergraph, run_mem0, run_readagent, run_single
from benchmarks.comparison_worker import atomic_json
from benchmarks.findingdory_structured import caption_config, fingerprint
from memory_agent.settings import Settings


@pytest.fixture
def workspace_dir():
    root = Path(__file__).resolve().parents[1] / "runtime" / "single_runner_tests" / uuid4().hex
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root)


def test_each_entrypoint_uses_its_own_method(monkeypatch):
    called = []
    for module in (run_hypergraph, run_mem0, run_amem, run_readagent):
        monkeypatch.setattr(module, "run_single",
                            lambda method, argv=None: called.append((method, argv)))
        module.main(["--run-dir", "test"])
    assert called == [(name, ["--run-dir", "test"]) for name in run_single.METHODS]


def test_four_independent_runs_share_one_verified_perception(workspace_dir, monkeypatch):
    data, shared = workspace_dir / "data", workspace_dir / "shared"
    manifest = {"episodes": ["ep_1", "ep_33"]}
    tasks = [{"row_idx": i} for i in range(12)]
    atomic_json(data / "manifest.json", manifest)
    monkeypatch.setattr(run_single.Settings, "load", classmethod(lambda cls: Settings(
        llm_key="fake", llm_model="fake", embedding_key="fake", embedding_model="fake")))
    monkeypatch.setattr(run_single, "environment_check", lambda methods: {})
    monkeypatch.setattr(run_single, "verify_data", lambda directory: (manifest, tasks))
    monkeypatch.setattr(run_single, "ensure_frames", lambda *args: None)
    calls = {"perception": 0, "methods": []}

    def fake_observe(root, *_):
        calls["perception"] += 1
        for ep in manifest["episodes"]:
            frames = [{"frame_index": i, "description": f"frame {i}", "entities": [],
                       "relations": [], "interaction": None, "uncertainty": []}
                      for i in range(96)]
            atomic_json(root / "observations" / f"{ep}.json", {
                "episode": ep, "schema_version": run_single.SCHEMA_VERSION,
                "observations": frames})
        atomic_json(root / "perception" / "api_budget.json", {
            "requests_started": 1, "reported_tokens": 10})

    def fake_execute(root, directory, method, plugins):
        calls["methods"].append(method)
        assert all((root / "observations" / f"{ep}.json").exists()
                   for ep in manifest["episodes"])
        for task in tasks:
            atomic_json(root / "methods" / method / "answers" / f"{task['row_idx']}.json", {
                "row_idx": task["row_idx"], "method": method, "correct": True,
                "candidate_frame_covered": 1.0})
        atomic_json(root / "methods" / method / "api_budget.json", {
            "requests_started": 12, "reported_tokens": 120})

    monkeypatch.setattr(run_single, "observe", fake_observe)
    monkeypatch.setattr(run_single, "execute_method", fake_execute)
    for method in run_single.METHODS:
        run_single.main(method, ["--run-dir", str(workspace_dir / method),
                                 "--shared-perception-dir", str(shared),
                                 "--data-dir", str(data)])
    assert calls == {"perception": 1, "methods": list(run_single.METHODS)}
    for method in run_single.METHODS:
        run = workspace_dir / method
        assert json.loads((run / "config.json").read_text(encoding="utf-8"))["methods"] == [method]
        assert json.loads((run / "summary.json").read_text(encoding="utf-8"))[0]["correct"] == 12
        assert "12/12" in (run / "REPORT.md").read_text(encoding="utf-8")
        assert (run / "perception_origin.json").exists()
        for ep in manifest["episodes"]:
            assert (run / "observations" / f"{ep}.json").read_bytes() == \
                   (shared / "observations" / f"{ep}.json").read_bytes()


def test_existing_run_cannot_mix_changed_shared_observations(workspace_dir):
    source, destination = workspace_dir / "shared.json", workspace_dir / "run.json"
    source.write_text("one", encoding="utf-8")
    run_single.copy_verified(source, destination)
    source.write_text("two", encoding="utf-8")
    with pytest.raises(RuntimeError, match="不一致"):
        run_single.copy_verified(source, destination)


def test_sealed_shared_observations_reject_later_edits(workspace_dir):
    shared = workspace_dir / "shared"
    source = shared / "observations" / "ep_1.json"
    atomic_json(source, {"observations": [1]})
    run_single.seal_observations(shared, {"episodes": ["ep_1"]})
    atomic_json(source, {"observations": [2]})
    with pytest.raises(RuntimeError, match="已变化"):
        run_single.seal_observations(shared, {"episodes": ["ep_1"]})


def test_complete_prior_perception_can_seed_without_model_requests(workspace_dir):
    data, source, shared = (workspace_dir / name for name in ("data", "source", "shared"))
    manifest = {"episodes": ["ep_1", "ep_33"]}
    tasks = [{"row_idx": i} for i in range(12)]
    settings = Settings(llm_key="fake", llm_model="fake")
    atomic_json(data / "manifest.json", manifest)
    prior_caption = caption_config(settings, data)
    prior_caption["validator_sha256"] = "legacy-whole-module-hash"
    atomic_json(source / "config.json", {
        "dataset": manifest, "selected_tasks_sha256": fingerprint(tasks),
        "caption": prior_caption, "llm_url": settings.llm_url,
        "vision_model": settings.vision_model, "llm_model": settings.llm_model})
    for ep in manifest["episodes"]:
        atomic_json(source / "observations" / f"{ep}.json", {
            "episode": ep, "schema_version": run_single.SCHEMA_VERSION,
            "observations": [{"frame_index": i, "description": f"frame {i}",
                              "entities": [], "relations": [], "interaction": None,
                              "uncertainty": []} for i in range(96)]})
    atomic_json(source / "perception" / "api_budget.json", {
        "requests_started": 117, "reported_tokens": 1353847})
    assert run_single.seed_from_comparison(source, shared, data, manifest, tasks, settings)
    assert run_single.complete_observations(shared, manifest)
    assert run_single.seal_observations(shared, manifest)
    assert (shared / "import_origin.json").exists()
