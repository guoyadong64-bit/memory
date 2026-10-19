"""The 50-video perception stage is tested without network or model calls."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from benchmarks import prepare_validation50_observations as stage
from benchmarks.comparison_worker import atomic_json
from benchmarks.findingdory_memory import VisualFrame
from memory_agent.settings import Settings


@pytest.fixture
def workspace_dir():
    base = Path(__file__).resolve().parents[1] / "runtime" / "validation50_tests"
    root = base / uuid4().hex
    root.mkdir(parents=True)
    yield root
    assert root.resolve().is_relative_to(base.resolve())
    shutil.rmtree(root)


def test_resume_seal_and_changed_output_rejected(workspace_dir, monkeypatch):
    data, shared = workspace_dir / "data", workspace_dir / "shared"
    data.mkdir()
    episodes = ["ep_1", "ep_2"]
    manifest = {"revision": "fixed-test-revision", "episodes": episodes,
                "videos": {f"videos/val/{ep}.mp4": {"sha256": ep}
                           for ep in episodes}}
    monkeypatch.setattr(stage, "selected_data", lambda _: manifest)
    monkeypatch.setattr(stage.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(stage, "executable_version", lambda _: "ffmpeg test")

    def fake_frames(_data, output, ep, *_args):
        path = output / "frames" / ep / "frames_manifest.json"
        if not path.exists():
            atomic_json(path, {"episode": ep})
        return stage.checksum(path)

    monkeypatch.setattr(stage, "ensure_frames", fake_frames)
    settings = Settings(llm_key="fake-key", llm_model="fake-vision")
    calls = {"count": 0, "fail": True}

    def extractor(client, _shared, _ep, indices, _previous, _diagnostic, **_kwargs):
        calls["count"] += 1
        if calls["fail"] and calls["count"] == 3:
            raise RuntimeError("injected interruption")
        client.budget.begin()
        client.budget.finish("visual_observation", "fake-vision", {"total_tokens": 7})
        return [VisualFrame(frame_index=index, description=f"frame {index}")
                for index in indices]

    with pytest.raises(RuntimeError, match="injected interruption"):
        stage.run(data, shared, settings, extractor=extractor)
    assert len(list((shared / "perception" / "batches" / "ep_1").glob("*.json"))) == 2
    assert not (shared / "observations_manifest.json").exists()

    calls["fail"] = False
    stage.run(data, shared, settings, extractor=extractor)
    assert calls["count"] == 97  # 96 batches plus one interrupted call
    assert len(list((shared / "observations").glob("*.json"))) == 2
    assert json.loads((shared / "perception" / "api_budget.json").read_text())["requests_started"] == 96
    sealed = json.loads((shared / "observations_manifest.json").read_text())
    assert list(sealed["observation_sha256"]) == episodes

    stage.run(data, shared, settings, extractor=extractor)
    assert calls["count"] == 97

    changed = shared / "observations" / "ep_1.json"
    output = json.loads(changed.read_text(encoding="utf-8"))
    output["observations"][0]["description"] = "changed"
    atomic_json(changed, output)
    with pytest.raises(RuntimeError, match="共享观测内容被改动"):
        stage.run(data, shared, settings, extractor=extractor)


def test_first50_data_selection_and_question_counts(workspace_dir):
    data = workspace_dir / "data"
    videos = [f"videos/val/ep_{index}.mp4" for index in range(1, 51)]
    counts = [58] * 49 + [59]
    manifest = {"dataset": "yali30/findingdory-subsampled-96", "split": "validation",
                "selected_questions": 2901, "videos": {}}
    rows = []
    for video, count in zip(videos, counts):
        path = data / video
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(video.encode())
        manifest["videos"][video] = {"questions": count, "bytes": path.stat().st_size,
                                      "sha256": stage.checksum(path)}
        first_index = len(rows)
        rows.extend({"row_idx": first_index + index, "video": video}
                    for index in range(count))
    atomic_json(data / "validation_first50_manifest.json", manifest)
    questions = data / "validation_first50_questions.jsonl"
    questions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    manifest = stage.selected_data(data)
    assert len(manifest["episodes"]) == 50
    assert manifest["selected_questions"] == 2901
    assert manifest["episodes"][0] == "ep_1"
    assert manifest["episodes"][-1] == "ep_50"
