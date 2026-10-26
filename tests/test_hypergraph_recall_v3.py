"""Offline invariants for bounded, provenance-preserving entity trace retrieval."""
import inspect
import json
from datetime import datetime, timedelta, timezone

from benchmarks.hypergraph_evidence_subgraph import record_frames
from benchmarks.hypergraph_recall_v3 import project_retrieval, query_subgraph_v3
from memory_agent.models import EntityLabel, MemoryEntity, Observation, Record, RoleBinding, SpatialContext


def frame_record(namespace, n, *, entity_id="cup_a", name="cup", description_size=50):
    """Each frame also sees a shared robot and support; those are retrieval hubs."""
    stamp = datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=n)
    entities = [
        MemoryEntity(id=entity_id, name=name, kind="object", namespace=namespace,
                     attributes={"role": "movable", "visibility": "clear"}),
        MemoryEntity(id="shared_table", name="table", kind="object", namespace=namespace,
                     attributes={"role": "support"}),
        MemoryEntity(id="shared_robot", name="robot", kind="agent", namespace=namespace),
    ]
    value = {"frame_index": n, "clock": f"06:{n // 60:02d}:{n % 60:02d}",
             "description": f"{name} is on the table. " + "x" * description_size,
             "entities": [{"id": e.id, "name": e.name, "kind": e.kind} for e in entities],
             "relations": [{"source": entity_id, "predicate": "on", "target": "shared_table",
                            "confidence": .95}], "interaction": None, "uncertainty": []}
    observation = Observation(id=f"{namespace}_image_{n}", source=f"{namespace}/{n}.jpg",
        modality="image", namespace=namespace, entity=entity_id, metric="visual_state",
        value=json.dumps(value), unit="description", observed_at=stamp,
        provenance="Synthetic witnessed image used only for offline tests")
    return Record(id=f"{namespace}_frame_{n}", key="history", question=f"{name} at frame {n}",
        answer=json.dumps(value), source=observation.source, observations=[observation], entities=entities,
        space=SpatialContext(namespace=namespace, entities=[e.id for e in entities]),
        role_bindings=[RoleBinding(role="visible", entity_id=e.id,
                                  observation_ids=[observation.id], confidence=.95) for e in entities])


def frame_set(records):
    return {n for record in records for n in record_frames(record)}


def assert_bounded(records, context, seed_frames):
    expansion = context["trace_expansion"]
    assert len(records) <= 20
    assert expansion["added_unique_frames"] <= 8
    assert expansion["promoted_entities"] <= 4
    assert expansion["input_unique_frames"] == len(seed_frames)
    assert expansion["output_unique_frames"] == len(frame_set(records))
    assert seed_frames <= frame_set(records)
    assert expansion["added_unique_frames"] == len(frame_set(records) - seed_frames)
    assert all(len(record_frames(record)) <= 4 for record in records if "_trace_" in record.id)


def test_query_entity_trace_expands_real_history_without_background_or_actor_flood():
    records = [frame_record("ep", n, entity_id="cup_a" if n in {0, 40, 95} else f"box_{n}",
                            name="cup" if n in {0, 40, 95} else "box") for n in range(96)]
    selected, context = query_subgraph_v3(records, [records[0].id], "Navigate to a cup.", {})
    assert_bounded(selected, context, {0})
    assert any("_trace_" in record.id for record in selected)
    assert 95 in frame_set(selected)
    # Shared background and actor must never provide a path to the other 93 images.
    assert frame_set(selected) <= {0, 40, 95}


def test_same_entity_id_in_another_namespace_does_not_join_a_seed_trace():
    records = [frame_record("ep_a", 0), frame_record("ep_a", 10), frame_record("ep_b", 80)]
    selected, context = query_subgraph_v3(records, [records[0].id], "Navigate to a cup.", {})
    assert_bounded(selected, context, {0})
    assert 10 in frame_set(selected)
    assert 80 not in frame_set(selected)
    assert all(observation.namespace == "ep_a" for record in selected for observation in record.observations)


def test_same_named_different_ids_are_not_merged_into_one_physical_trace():
    records = [frame_record("ep", 0), frame_record("ep", 5)] + [
        frame_record("ep", n, entity_id=f"other_cup_{n}") for n in range(10, 50)]
    selected, context = query_subgraph_v3(records, [records[0].id], "Navigate to a cup.", {})
    assert_bounded(selected, context, {0})
    assert 5 in frame_set(selected)
    traces = [record for record in selected if "_trace_" in record.id]
    assert traces
    for trace in traces:
        witnessed_ids = {
            json.loads(observation.value)["entities"][0]["id"] for observation in trace.observations
        }
        assert len(witnessed_ids) == 1


def test_unique_long_seed_observation_set_cannot_be_lost_by_four_frame_trace_cap():
    source = [frame_record("ep", n) for n in range(7)]
    long_seed = source[0].model_copy(update={
        "id": "ep_long_seed", "observations": [record.observations[0] for record in source],
        "role_bindings": [RoleBinding(role="visible", entity_id=entity.id,
            observation_ids=[record.observations[0].id for record in source], confidence=.95)
            for entity in source[0].entities],
    })
    records = [long_seed, *source, frame_record("ep", 30)]
    selected, context = query_subgraph_v3(records, [long_seed.id], "Navigate to a cup.", {})
    assert_bounded(selected, context, set(range(7)))
    assert set(range(7)) <= frame_set(selected)


def test_trace_observations_are_original_witnesses_and_input_records_are_unchanged():
    records = [frame_record("ep", n) for n in (0, 5, 15, 25, 35, 45)]
    originals = {observation.id: observation for record in records for observation in record.observations}
    before = [record.model_dump(mode="json") for record in records]
    selected, context = query_subgraph_v3(records, [records[0].id], "Navigate to a cup.", {})
    assert_bounded(selected, context, {0})
    assert any("_trace_" in record.id for record in selected)
    for record in selected:
        for observation in record.observations:
            assert observation.id in originals
            assert observation is originals[observation.id]
            assert json.loads(observation.value)["frame_index"] in record_frames(record)
    assert before == [record.model_dump(mode="json") for record in records]


def test_catalog_only_identity_cannot_import_an_unwitnessed_frame():
    seed = frame_record("ep", 0)
    actual_later_view = frame_record("ep", 20)
    unrelated = frame_record("ep", 10, entity_id="box_a", name="box")
    # A process may catalog an entity that is not visible in every component frame.
    unrelated.entities.append(seed.entities[0])
    selected, context = query_subgraph_v3([seed, unrelated, actual_later_view], [seed.id],
                                         "Navigate to a cup.", {})
    assert_bounded(selected, context, {0})
    assert 20 in frame_set(selected)
    assert 10 not in frame_set(selected)


def test_projection_keeps_byte_budget_and_cannot_claim_unserialized_frames():
    records = [frame_record("ep", n, description_size=3000) for n in range(20)]
    evidence, resolution, audit = project_retrieval(records, [record.id for record in records[:16]],
                                                   "Navigate to a cup.", {})
    assert len(evidence) <= 20
    assert len(json.dumps(evidence, ensure_ascii=False).encode("utf-8")) <= 40000
    original_frames = frame_set(records)
    output_frames = {n for item in evidence for n in item["frame_indices"]}
    serialized_frames = {row[0] for item in evidence for row in item["content"]["frame_rows"]}
    assert output_frames <= original_frames
    assert output_frames == serialized_frames
    assert audit["bytes"] <= 40000
    # Gold data must never become an input to either projection entry point.
    assert not any("gold" in name for name in inspect.signature(query_subgraph_v3).parameters)
    assert not any("gold" in name for name in inspect.signature(project_retrieval).parameters)


def test_budget_pressure_cannot_evict_previously_retained_seed_witnesses():
    from benchmarks.hypergraph_recall_v2 import project_retrieval as project_v2

    records = []
    for n in [*range(20), 30, 40]:
        record = frame_record("ep", n)
        observation = record.observations[0]
        frame = json.loads(observation.value)
        # The packer deliberately retains uncertainty verbatim. Expansion must
        # not buy more candidate views by removing the old bounded evidence.
        frame["uncertainty"] = ["x" * 2500]
        value = json.dumps(frame)
        records.append(record.model_copy(update={"answer": value, "observations": [
            observation.model_copy(update={"value": value})]}))
    seed_ids = [record.id for record in records[:20]]
    baseline, _, _ = project_v2(records, seed_ids, "Navigate to a cup.", {})
    evidence, _, audit = project_retrieval(records, seed_ids, "Navigate to a cup.", {})
    baseline_frames = {n for item in baseline for n in item["frame_indices"]}
    final_frames = {n for item in evidence for n in item["frame_indices"]}
    assert baseline_frames <= final_frames
    assert len(json.dumps(evidence, ensure_ascii=False).encode("utf-8")) <= 40000
    assert audit["bytes"] <= 40000


def test_named_object_can_expand_its_witnessed_source_location_without_new_role_claims():
    seed = frame_record("ep", 0)
    observation = seed.observations[0]
    seed.role_bindings.extend([
        RoleBinding(role="object", entity_id="cup_a", observation_ids=[observation.id], confidence=.95),
        RoleBinding(role="source", entity_id="shared_table", observation_ids=[observation.id], confidence=.95),
    ])
    records = [seed, frame_record("ep", 10, entity_id="box_a", name="box"),
               frame_record("ep", 20, entity_id="box_b", name="box")]
    question = "Navigate to the receptacle you picked a cup from."
    selected, context = query_subgraph_v3(records, [seed.id], question, {})
    assert_bounded(selected, context, {0})
    source_traces = [record for record in selected if "_trace_" in record.id and
                     json.loads(record.answer)["entity_id"] == "shared_table"]
    assert source_traces
    assert 20 in frame_set(source_traces)
    # The location trace may carry the observed source role, but cannot invent
    # source-role evidence in the later pictures which only show visibility.
    source_roles = [role for record in source_traces for role in record.role_bindings if role.role == "source"]
    assert source_roles
    assert all(set(role.observation_ids) == {observation.id} for role in source_roles)
    assert all(json.loads(record.answer)["continuous_event"] is False for record in source_traces)


def test_only_high_confidence_existing_labels_can_match_an_otherwise_unrelated_identity():
    for confidence in (.79, .8):
        records = [frame_record("ep", 0), frame_record("ep", 5)]
        for n in (80, 90):
            record = frame_record("ep", n, entity_id="ambiguous_item", name="item")
            observation = record.observations[0]
            frame = json.loads(observation.value)
            frame["entities"][0]["labels"] = [{"text": "cup", "confidence": confidence}]
            value = json.dumps(frame)
            entities = [record.entities[0].model_copy(update={"labels": [EntityLabel(
                text="cup", confidence=confidence, observation_ids=[observation.id])]}), *record.entities[1:]]
            records.append(record.model_copy(update={"entities": entities, "answer": value,
                "observations": [observation.model_copy(update={"value": value})]}))
        selected, context = query_subgraph_v3(records, [records[0].id], "Navigate to a cup.", {})
        assert_bounded(selected, context, {0})
        if confidence < .8:
            assert not frame_set(selected) & {80, 90}
        else:
            assert frame_set(selected) & {80, 90}
