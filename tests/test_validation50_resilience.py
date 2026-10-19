"""No model or network calls: structural fallback cannot create visual facts."""
import json
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from benchmarks.resume_validation50_observations import _live_extractor
from benchmarks.findingdory_memory import VisualFrame, validate_batch
from benchmarks.validation50_resilience import normalize_observations
from memory_agent.settings import Settings


@pytest.fixture
def workspace_dir():
    root = Path(__file__).resolve().parents[1] / "runtime" / "validation50_tests" / uuid4().hex
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root)


def _frame(index, object_id, interaction_id):
    return {"frame_index": index, "description": "The robot holds the visible object.",
            "clock": None,
            "entities": [{"id": object_id, "name": object_id.replace("_", " "),
                          "kind": "object", "labels": [], "attributes": {}},
                         {"id": "robot", "name": "robot", "kind": "agent",
                          "labels": [], "attributes": {}}],
            "relations": [{"source": object_id, "predicate": "held_by",
                           "target": "robot", "confidence": .95}],
            "interaction": {"id": interaction_id, "object_id": object_id,
                            "actor_id": "robot", "phase": "transport", "action": "carry",
                            "source_id": None, "target_id": None, "outcome": None,
                            "confidence": .9}, "uncertainty": []}


def test_reused_interaction_id_is_separated_without_changing_visual_claim():
    previous = [VisualFrame.model_validate(_frame(0, "red_cup", "int_pickup"))]
    raw = _frame(1, "brown_pot", "int_pickup")
    normalized, changes = normalize_observations({"observations": [raw]}, [1], previous)
    result = validate_batch(normalized["observations"], [1], previous)[0]
    assert result.interaction.object_id == "brown_pot"
    assert result.interaction.id != "int_pickup"
    assert result.description == raw["description"]
    assert result.relations[0].predicate == "held_by"
    assert [change["field"] for change in changes] == ["interaction.id"]


def test_undeclared_object_drops_phase_and_preserves_description():
    raw = _frame(1, "brown_pot", "int_pickup")
    raw["entities"] = [entity for entity in raw["entities"] if entity["id"] != "brown_pot"]
    normalized, changes = normalize_observations({"observations": [raw]}, [1], [])
    result = validate_batch(normalized["observations"], [1])[0]
    assert result.interaction is None
    assert result.description == raw["description"]
    assert result.relations == []
    assert result.uncertainty
    assert {change["field"] for change in changes} == {"interaction", "relations"}


def test_missing_frame_fails_instead_of_fabricating_observation():
    try:
        normalize_observations({"observations": []}, [1], [])
    except ValueError as exc:
        assert "omitted" in str(exc)
    else:
        raise AssertionError("Missing frame must not be accepted")


def test_live_extractor_accepts_id_repair_on_first_mock_response(workspace_dir):
    tmp_path = workspace_dir
    frame_dir = tmp_path / "frames" / "ep_1"
    frame_dir.mkdir(parents=True)
    for index in (0, 1):
        (frame_dir / f"{index:03d}.jpg").write_bytes(b"mock-image")

    class FakeClient:
        settings = Settings(llm_key="fake", llm_model="fake-vision")

        def __init__(self):
            self.calls = []

        def post(self, _url, _key, _payload, _role):
            self.calls.append({"model": "fake-vision"})
            return {"choices": [{"message": {"content": json.dumps({
                "observations": [_frame(1, "brown_pot", "int_pickup")]})}}]}

    client = FakeClient()
    prior = [VisualFrame.model_validate(_frame(0, "red_cup", "int_pickup"))]
    diagnostic = tmp_path / "perception" / "validation_checks" / "ep_1" / "001.json"
    frames = _live_extractor(client, tmp_path, "ep_1", [1], prior, diagnostic,
                             max_attempts=2)
    assert len(client.calls) == 1
    assert frames[0].interaction.id != "int_pickup"
    assert len(list((tmp_path / "perception" / "normalizations" /
                     "ep_1").glob("*.json"))) == 1
