import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
from uuid import uuid4

import httpx

from benchmarks.hypergraph_evidence_subgraph import pack_subgraph
from benchmarks.hypergraph_recall_v2 import project_retrieval, query_subgraph
from memory_agent.models import Evolution, MemoryEntity, Observation, Record, RoleBinding, Stage
from memory_agent.clients import ModelClients
from memory_agent.settings import Settings


def make_record(identity, indices, *, process=False, object_id="cup", description_size=1500):
    stamp = datetime(2020, 1, 1, tzinfo=timezone.utc)
    observations = []
    for n in indices:
        frame = {"frame_index": n, "clock": f"06:{n:02d}",
                 "description": "Visible object and support. " + "x" * description_size,
                 "entities": [{"id": object_id, "name": object_id, "kind": "object"},
                              {"id": "table", "name": "table", "kind": "place"}],
                 "relations": [{"source": object_id, "predicate": "on", "target": "table", "confidence": .9}],
                 "interaction": None, "uncertainty": ["Other actions may be unobserved."]}
        observations.append(Observation(id=f"obs_{n}", source="image", modality="image", namespace="ep",
            entity=object_id, provenance="synthetic image observation",
            metric="visual_state", value=json.dumps(frame), unit="description", observed_at=stamp + timedelta(seconds=n)))
    entities = [MemoryEntity(id=object_id, name=object_id, kind="object", namespace="ep"),
                MemoryEntity(id="table", name="table", kind="place", namespace="ep")]
    roles = [RoleBinding(role="visible", entity_id=object_id, observation_ids=[o.id for o in observations], confidence=.9)]
    evolution = None
    if process:
        stages = [Stage(id=f"s_{n}", phase="transport", at=stamp + timedelta(seconds=n),
                        state={"object_id": object_id}, observation_ids=[f"obs_{n}"]) for n in indices]
        evolution = Evolution(namespace="ep", case_id=identity, stages=stages)
        roles.append(RoleBinding(role="object", entity_id=object_id, observation_ids=[o.id for o in observations],
                                 stage_id=stages[0].id, confidence=.9))
    return Record(id=identity, key="history", question="historical observations", answer="{}", source="image",
                  observations=observations, entities=entities, role_bindings=roles, evolution=evolution)


def test_shared_projection_preserves_witness_frames_and_relations_under_budget():
    records = [make_record("event", list(range(10)), process=True)] + [
        make_record(f"frame_{i}", [i]) for i in range(10)]
    evidence, _, audit = pack_subgraph(records, max_bytes=7000)
    assert len(json.dumps(evidence, ensure_ascii=False).encode()) <= 7000
    assert audit["dropped_record_ids"] == []
    assert audit["input_unique_frames"] == audit["output_unique_frames"] == 10
    # Shared frame text appears once, while both memories retain real provenance.
    rows = [row for item in evidence for row in item["content"]["frame_rows"]]
    assert len(rows) == 10
    assert {row[0] for row in rows} == set(range(10))
    assert all(row[4] for row in rows)
    assert evidence[1]["frame_indices"] == [0]
    assert evidence[0]["content"]["trajectory"]


def test_small_budget_does_not_claim_dropped_target_or_anchor():
    record = make_record("event", [0], process=True)
    resolution = {"targets": [{"record_id": "event", "frame_index": 0}],
                  "anchor_record_ids": ["event"], "complete": True, "warnings": []}
    evidence, revised, audit = pack_subgraph([record], resolution, max_bytes=2)
    assert evidence == [] and audit["bytes"] == 2
    assert not revised["complete"] and revised["targets"] == []
    assert revised["warnings"]


def test_graph_neighbors_are_uncertain_and_do_not_merge_different_objects():
    events = [make_record(f"event_{n}", [n], process=True, object_id=f"object_{n}") for n in range(4)]
    selected, context = query_subgraph(events, ["event_0"],
        "Navigate to the object immediately after ending the interaction with object 1.", {})
    assert "event_1" in {record.id for record in selected}
    assert "event_2" in {record.id for record in selected}
    assert all(not edge["complete_history"] for edge in context["event_links"])
    evidence, revised, audit = project_retrieval(events, ["event_0"],
        "Navigate to the object immediately after ending the interaction with object 1.", {"complete": False})
    assert not revised["complete"]  # Neighbors never certify a guessed temporal answer.
    assert len(evidence[0]["content"]["entity_catalog"]) >= 4
    assert audit["retained_records"] <= 20


def test_export_does_not_extend_any_records_frame_provenance():
    records = [make_record("a", [1, 3]), make_record("b", [3, 5])]
    evidence, _, _ = pack_subgraph(records)
    assert {item["id"]: item["frame_indices"] for item in evidence} == {"a": [1, 3], "b": [3, 5]}
    rows = [row for item in evidence for row in item["content"]["frame_rows"]]
    assert sorted(row[0] for row in rows) == [1, 3, 5]


def test_graph_expansion_cannot_evict_unique_seed_frames_at_fixed_depth():
    seeds = [make_record(f"seed_{i}", [i], process=True, object_id=f"thing_{i}") for i in range(20)]
    neighbor = make_record("neighbor", [25], process=True, object_id="special_anchor")
    selected, _ = query_subgraph([*seeds, neighbor], [r.id for r in seeds],
        "Navigate to the object immediately after ending the interaction with special anchor.", {})
    assert len(selected) == 20
    assert {n for record in selected for n in record_frames_for_test(record)} >= set(range(20))


def record_frames_for_test(record):
    return [json.loads(o.value)["frame_index"] for o in record.observations]


def test_new_subgraph_works_with_unchanged_visual_answerer_without_gold():
    from benchmarks.findingdory_visual_answer import decode_visual
    root = Path(__file__).resolve().parents[1] / "runtime/subgraph_transport_tests" / uuid4().hex
    image = root / "frames/ep/000.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake-image-only-used-by-mock-transport")
    record = make_record("fact", [0])
    evidence, resolution, _ = pack_subgraph([record], {})
    seen = []

    def transport(request):
        payload = json.loads(request.content)
        text = json.loads(payload["messages"][1]["content"][0]["text"])
        assert "gold_goal_frames" not in text and "answer" not in text
        assert text["evidence"][0]["content"]["frame_rows"]
        images = [x for x in payload["messages"][1]["content"] if x["type"] == "image_url"]
        assert len(images) == 1
        seen.append(payload)
        return httpx.Response(200, json={"model": "fake", "choices": [{"message": {"content": json.dumps({
            "goal_frames": [0], "evidence_ids": ["fact"], "target_bindings": []})}}]})

    client = ModelClients(Settings(llm_key="fake", llm_model="fake"), transport=httpx.MockTransport(transport))
    try:
        output = decode_visual(client, "Navigate to a cup.", evidence,
            [json.loads(record.observations[0].value)], root, "ep", resolution)
        assert output["goal_frames"] == [0] and not output["repairs"]
        assert len(seen) == 1
    finally:
        client.close()
        shutil.rmtree(root)
