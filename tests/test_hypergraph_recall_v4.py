"""Offline safety and evidence-budget checks for uncertain boundary recall."""
import inspect
import json
from datetime import datetime, timedelta, timezone

import pytest

from benchmarks.hypergraph_evidence_subgraph import record_frames
from benchmarks.hypergraph_recall_v3 import project_retrieval as project_v3
from benchmarks.hypergraph_recall_v4 import project_retrieval, query_subgraph_v4
from memory_agent.models import (EntityLabel, Evolution, MemoryEntity, Observation, Record,
                                 RoleBinding, SpatialContext, Stage)


def frame_record(namespace, n, *, identity="cup", phase=None, uncertainty_size=0):
    stamp = datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=n)
    observation_id = f"{namespace}_observation_{n}"
    entity = MemoryEntity(id=identity, namespace=namespace, name=identity, kind="object",
        labels=[EntityLabel(text=identity, confidence=.95, observation_ids=[observation_id])])
    frame = {"frame_index": n, "clock": f"06:{n // 60:02d}:{n % 60:02d}",
        "description": f"A witnessed view of {identity}.",
        "entities": [{"id": identity, "name": identity, "kind": "object"}],
        "relations": [], "uncertainty": ["x" * uncertainty_size] if uncertainty_size else [],
        "interaction": ({"object_id": identity, "phase": phase, "confidence": .95,
                         "actor_id": None, "source_id": None, "target_id": None,
                         "action": phase, "outcome": None} if phase else None)}
    value = json.dumps(frame)
    observation = Observation(id=observation_id, namespace=namespace, source=f"{namespace}/{n}.jpg",
        modality="image", entity=identity, metric="visual_state", unit="description", value=value,
        observed_at=stamp, provenance="Synthetic original observation for offline invariant tests")
    return Record(id=f"{namespace}_frame_{n}", key="history", question=f"View of {identity}",
        answer=value, source=observation.source, observations=[observation], entities=[entity],
        space=SpatialContext(namespace=namespace, entities=[identity]),
        role_bindings=[RoleBinding(role="visible", entity_id=identity,
                                  observation_ids=[observation_id], confidence=.95)])


def frames(records):
    return {n for record in records for n in record_frames(record)}


def evidence_frames(evidence):
    return {n for item in evidence for n in item["frame_indices"]}


def test_incomplete_boundaries_are_candidates_without_dropping_seed_observations():
    records = [frame_record("ep", n, identity=f"object_{n}", phase="approach") for n in range(15)]
    chosen, _, context, _ = query_subgraph_v4(records, [records[0].id],
        "Navigate to the second object that you interacted with yesterday.", {})
    assert 0 in frames(chosen)
    assert 1 < len(frames(chosen)) <= 9
    assert len(chosen) <= 20
    assert len(frames(chosen) - {0}) <= 8
    assert context["context_expansion"]["added_unique_frames"] <= 8
    assert context["context_expansion"]["edges_are_hypotheses"] is True


def test_context_recall_cannot_join_another_episode_with_the_same_identity():
    records = [frame_record("ep_a", 0), frame_record("ep_a", 1),
               frame_record("ep_b", 50, phase="approach")]
    chosen, _, _, _ = query_subgraph_v4(records, [records[0].id],
        "Navigate to the second object that you interacted with yesterday.", {})
    assert 50 not in frames(chosen)
    assert all(o.namespace == "ep_a" for r in chosen for o in r.observations)


@pytest.mark.parametrize("phase", ["approach", "settled"])
def test_partial_phase_is_not_promoted_to_a_certified_interaction(phase):
    records = [frame_record("ep", 0), frame_record("ep", 10, identity="bowl", phase=phase)]
    chosen, _, context, metadata = query_subgraph_v4(records, [records[0].id],
        "Navigate to the second object that you interacted with yesterday.", {})
    assert 10 in frames(chosen)
    assert not context["observed_event_fragments"]
    context_views = [r for r in chosen if r.id in metadata]
    assert context_views
    for view in context_views:
        projection = metadata[view.id]
        assert projection["certified_event_membership"] is False
        assert view.evolution is None
        assert not any(role.role in {"source", "destination", "object"} for role in view.role_bindings)
        original_phase = json.loads(next(o.value for o in view.observations
                                        if json.loads(o.value)["frame_index"] == 10))["interaction"]["phase"]
        assert original_phase == phase


def test_views_reference_original_observations_and_pass_full_model_validation():
    records = [frame_record("ep", n, identity=f"object_{n}", phase="approach") for n in range(12)]
    originals = {o.id: o for record in records for o in record.observations}
    before = [record.model_dump(mode="json") for record in records]
    chosen, _, _, _ = query_subgraph_v4(records, [records[0].id],
        "Navigate to the second object that you interacted with yesterday.", {})
    for view in chosen:
        Record.model_validate(view.model_dump(mode="json"))
        for observation in view.observations:
            assert observation is originals[observation.id]
        observation_ids = {o.id for o in view.observations}
        assert all(set(role.observation_ids) <= observation_ids for role in view.role_bindings)
        assert all(set(label.observation_ids) <= observation_ids
                   for entity in view.entities for label in entity.labels)
    assert before == [record.model_dump(mode="json") for record in records]


def test_projection_preserves_v3_frames_and_stays_within_bytes_and_record_budgets():
    records = [frame_record("ep", n, identity=f"object_{n}", phase="approach") for n in range(25)]
    seeds = [r.id for r in records[:20]]
    question = "Navigate to the second object that you interacted with yesterday."
    old, _, _ = project_v3(records, seeds, question, {})
    evidence, _, audit = project_retrieval(records, seeds, question, {})
    assert evidence_frames(old) <= evidence_frames(evidence)
    assert len(evidence) <= 20
    assert len(evidence_frames(evidence) - evidence_frames(old)) <= 8
    assert len(json.dumps(evidence, ensure_ascii=False).encode()) <= 40000
    assert audit["bytes"] <= 40000
    rows = {row[0] for item in evidence for row in item["content"]["frame_rows"]}
    assert rows == evidence_frames(evidence)


def test_budget_pressure_preserves_all_previously_serialized_v3_witnesses():
    records = [frame_record("ep", n, identity=f"object_{n}", phase="approach", uncertainty_size=2500)
               for n in range(24)]
    seeds = [r.id for r in records[:20]]
    question = "Navigate to the second object that you interacted with yesterday."
    old, _, _ = project_v3(records, seeds, question, {})
    evidence, _, _ = project_retrieval(records, seeds, question, {})
    assert evidence_frames(old) <= evidence_frames(evidence)
    assert len(json.dumps(evidence, ensure_ascii=False).encode()) <= 40000


def test_target_record_references_are_remapped_without_changing_target_proof():
    records = [frame_record("ep", 10), frame_record("ep", 11, identity="bowl", phase="approach")]
    original = records[0]
    resolution = {"record_ids": [original.id], "target_record_ids": [original.id],
        "anchor_record_ids": [original.id], "targets": [{"record_id": original.id,
            "entity_id": "cup", "role": "visible", "frame_index": 10,
            "observation_ids": [original.observations[0].id]}], "complete": True}
    evidence, revised, _ = project_retrieval(records, [original.id],
        "Navigate to the second object that you interacted with yesterday.", resolution)
    selected_ids = {item["id"] for item in evidence}
    target = revised["targets"][0]
    assert target["record_id"] in selected_ids
    assert target["frame_index"] == 10
    assert target["observation_ids"] == [original.observations[0].id]
    assert set(revised["anchor_record_ids"]) <= selected_ids


def test_retrieval_entry_points_accept_no_benchmark_labels():
    for function in (query_subgraph_v4, project_retrieval):
        assert not any("gold" in key or "category" in key or "row_idx" in key
                       for key in inspect.signature(function).parameters)


def test_background_query_object_cannot_ground_an_unrelated_action_boundary():
    records = [frame_record("ep", n) for n in (0, 20, 30, 40)]
    unrelated = frame_record("ep", 10, identity="ball", phase="approach")
    observation = unrelated.observations[0]
    frame = json.loads(observation.value)
    # The cup really is visible, but is neither the manipulated ball nor an
    # evidenced source/target/support participant in the ball's approach.
    frame["entities"].append({"id": "cup", "name": "cup", "kind": "object"})
    value = json.dumps(frame)
    unrelated = unrelated.model_copy(update={"answer": value,
        "observations": [observation.model_copy(update={"value": value})],
        "entities": [*unrelated.entities,
            MemoryEntity(id="cup", namespace="ep", name="cup", kind="object")],
        "role_bindings": [*unrelated.role_bindings,
            RoleBinding(role="visible", entity_id="cup", confidence=.95,
                        observation_ids=[observation.id])]})
    records.append(unrelated)
    old, _, _ = project_v3(records, [records[0].id], "Navigate to a cup.", {})
    assert 10 not in evidence_frames(old)
    chosen, _, _, metadata = query_subgraph_v4(records, [records[0].id], "Navigate to a cup.", {})
    assert 10 not in frames(chosen)
    assert not any(source["frame_index"] == 10 and source["edge"] == "incomplete_action_boundary"
                   for projection in metadata.values()
                   for source in projection.get("source_observations", []))


def test_event_context_serializes_root_trajectory_without_grouping_context_as_the_event():
    root = frame_record("ep", 10, phase="grasp")
    observation = root.observations[0]
    stage = Stage(id="observed_grasp", phase="grasp", at=observation.observed_at,
        state={"object_id": "cup"}, observation_ids=[observation.id])
    root = root.model_copy(update={"evolution": Evolution(namespace="ep", case_id="cup_event", stages=[stage]),
        "role_bindings": [*root.role_bindings, RoleBinding(role="object", entity_id="cup",
            observation_ids=[observation.id], stage_id=stage.id, confidence=.95)]})
    Record.model_validate(root.model_dump(mode="json"))
    records = [root, frame_record("ep", 9, identity="background_a"),
               frame_record("ep", 11, identity="background_b")]
    evidence, _, _ = project_retrieval(records, [root.id], "Navigate to a cup.", {})
    contexts = [item for item in evidence
                if item["content"].get("retrieval_projection", {}).get("kind") == "event_boundary_context_view"]
    assert contexts
    content = contexts[0]["content"]
    assert set(contexts[0]["frame_indices"]) == {9, 10, 11}
    # The unchanged common image selector treats a truthy `trajectory` as one
    # event containing EVERY item frame. Independent context must avoid it.
    assert not content.get("trajectory")
    assert content["root_event_frame_indices"] == [10]
    assert content["root_event_trajectory"] == [{"phase": "grasp", "witness_frames": [10]}]
    assert content["retrieval_projection"]["certified_event_membership"] is False


def test_exhausted_expansion_budget_preserves_existing_entity_trace_provenance(monkeypatch):
    from benchmarks import hypergraph_recall_v4 as module

    records = [frame_record("ep", n) for n in range(9)]
    root = records[0]
    trace_data = {"kind": "same_entity_observation_trace", "entity_id": "cup", "namespace": "ep",
        "continuous_event": False, "frame_indices": list(range(9)),
        "source_observations": [{"frame_index": n, "observation_id": records[n].observations[0].id,
                                 "source_record_id": records[n].id} for n in range(9)]}
    trace = root.model_copy(update={"id": "entity_trace_existing", "answer": json.dumps(trace_data),
        "observations": [r.observations[0] for r in records]})
    context = {"seed_projection_record_ids": [root.id], "candidate_sources": {},
        "trace_expansion": {"namespace": "ep", "added_unique_frames": 8,
            "input_unique_frames": 1, "output_unique_frames": 9,
            "promoted_entities": 1, "entity_ids": ["cup"]},
        "observed_event_fragments": [], "event_links": [], "caveat": "No certified continuous event."}
    monkeypatch.setattr(module, "query_subgraph_v3", lambda *args: ([trace], context))
    evidence, _, _ = module.project_retrieval(records, [root.id], "Navigate to a cup.", {})
    item = next(item for item in evidence if item["id"] == trace.id)
    assert item["content"]["retrieval_projection"] == trace_data
    assert set(item["frame_indices"]) == set(range(9))
