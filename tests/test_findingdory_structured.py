from datetime import timedelta
import json
from pathlib import Path
import shutil
from uuid import uuid4

import pytest
import httpx

from benchmarks.findingdory_memory import VisualFrame, frame_times, history_query, structured_records, validate_batch
from benchmarks.findingdory_structured import (FrameAnswerAgent, extract_batch, freeze,
    persist_records, planned_goal_answer)
from memory_agent.clients import ModelClients, ModelError
from memory_agent.settings import Settings
from memory_agent.memory import HypergraphMemory
from memory_agent.models import Query, RecallPlan
from memory_agent.recall import hint_score, record_match


@pytest.fixture
def directory():
    root = Path(__file__).resolve().parents[1] / "runtime" / "structured_tests" / uuid4().hex
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root)


def frame(index=0, clock="06:00", phase="grasp", **updates):
    data = {"frame_index": index, "description": "Robot transfers a cup between two surfaces.",
        "clock": clock, "entities": [{"id": "cup_1", "name": "white cup", "kind": "object"},
        {"id": "gripper_1", "name": "robot gripper", "kind": "agent"},
        {"id": "desk_1", "name": "wooden desk", "kind": "place"},
        {"id": "box_1", "name": "woven box", "kind": "place"}],
        "relations": [{"source": "cup_1", "predicate": "held_by", "target": "gripper_1", "confidence": .9}],
        "interaction": {"id": "move_cup_1", "object_id": "cup_1", "actor_id": "gripper_1", "phase": phase,
        "action": {"grasp": "pick_up", "transport": "carry", "release": "place", "settled": None}[phase],
        "source_id": "desk_1", "target_id": "box_1", "outcome": "cup on box" if phase == "settled" else None,
        "confidence": .9}, "uncertainty": []}
    if phase in {"release", "settled"}:
        data["relations"] = [{"source": "cup_1", "predicate": "on", "target": "box_1", "confidence": .9}]
    return data | updates


def history():
    return [frame(), frame(1, "06:01", "release"), frame(2, "06:02", "settled")]


def test_image_evidence_and_transitions_survive_sqlite_reload(directory):
    records = structured_records("ep_1", history())
    store = persist_records(directory, "ep_1", records, "hypergraph")
    process = next(r for r in records if r.evolution)
    loaded = HypergraphMemory(store.path)
    assert loaded.get(process.id).evolution == process.evolution
    assert len(loaded.trace("ep_1", process.evolution.case_id)[0]["observations"]) == 3
    assert {"spatial_relation", "stage_evidence", "state_transition", "process"} <= {
        e.kind for e in loaded.graph.edges.values()}
    assert records[0].time.event_at.hour == 6
    assert records[-1].evolution.stages[-1].phase == "settled"
    assert records[-1].outcome == "cup on box"


@pytest.mark.parametrize("strategy", ["flat", "hypergraph"])
def test_recall_reads_past_relations_and_process_but_application_keeps_gates(directory, strategy):
    records = structured_records("ep_1", history())
    store = persist_records(directory, "ep_1", records, strategy)
    recall = history_query("Where did the robot place the cup?", "ep_1", records)
    for r in records:
        assert store.eligible(r, recall)
    application = recall.model_copy(update={"purpose": "application"})
    assert not store.eligible(records[0], application)  # old spatial fact is not a current prerequisite
    assert not store.eligible(records[-1], application)  # missing initial state and phase
    wrong = history_query("Where is the cup?", "ep_2", records)
    assert not any(store.eligible(r, wrong) for r in records)
    future = recall.model_copy(update={"at": records[0].time.event_at - timedelta(seconds=1)})
    assert not any(store.eligible(r, future) for r in records)


def test_missing_clock_and_midnight_are_explicit_not_fabricated():
    raw = [frame(clock="23:59"), frame(1, "00:01"), frame(2, None)]
    times = frame_times([VisualFrame.model_validate(r) for r in raw])
    assert times[1][0].day == 2
    assert times[2][1] == "frame_sequence"
    records = structured_records("ep_1", raw)
    assert all(len({json.loads(o.value)["time_basis"] for o in r.observations}) == 1
               for r in records if r.evolution)  # partial singleton remains on its own time axis
    assert json.loads(next(r for r in records if r.id=="ep_1_frame_2").answer)["clock"] is None


def test_identity_conflicts_and_impossible_action_phase_are_rejected():
    prior = [VisualFrame.model_validate(frame())]
    renamed = frame(1)
    renamed["entities"][0]["name"] = "different object"
    with pytest.raises(ValueError, match="Stable entity"):
        validate_batch([renamed], [1], prior)
    invalid = frame(1, phase="settled")
    invalid["interaction"]["action"] = "pick_up"
    with pytest.raises(ValueError, match="contradicts phase"):
        VisualFrame.model_validate(invalid)
    bad = frame()
    bad["relations"][0]["target"] = "unseen_place"
    with pytest.raises(ValueError, match="declared endpoints"):
        VisualFrame.model_validate(bad)
    reverse = frame()
    reverse["relations"][0]["source"], reverse["relations"][0]["target"] = "gripper_1", "cup_1"
    with pytest.raises(ValueError, match="OBJECT -> AGENT"):
        VisualFrame.model_validate(reverse)
    missing = frame(description="The robot is carrying the cup.", interaction=None)
    incomplete = VisualFrame.model_validate(missing)
    assert incomplete.interaction is None
    assert incomplete.relations[0].predicate == "held_by"
    impossible = frame(phase="settled")
    impossible["relations"] = frame()["relations"]
    with pytest.raises(ValueError, match="cannot still be held_by"):
        VisualFrame.model_validate(impossible)


def test_explicit_manipulated_object_appearance_conflict_is_rejected():
    wrong = frame(description="The gripper grasps a purple cylinder object (cup_1).")
    wrong["entities"][0]["name"] = "black_white_toy"
    with pytest.raises(ValueError, match="Appearance conflict"):
        VisualFrame.model_validate(wrong)
    implicit = frame(description="The gripper is actively grasping a purple cylindrical object.")
    implicit["entities"][0]["name"] = "black_white_toy_penguin"
    with pytest.raises(ValueError, match="Appearance conflict"):
        VisualFrame.model_validate(implicit)
    grip = frame(description="The gripper fingers are closed around a small purple cylindrical object.")
    grip["entities"][0]["name"] = "black_white_toy_penguin"
    with pytest.raises(ValueError, match="Appearance conflict"):
        VisualFrame.model_validate(grip)
    unrelated = frame(description="The gripper grasps cup_1 near a purple rug.")
    unrelated["entities"][0]["name"] = "black_white_toy"
    assert VisualFrame.model_validate(unrelated).interaction.object_id == "cup_1"


def test_low_confidence_and_backwards_phases_do_not_create_false_processes():
    uncertain = history()
    for f in uncertain:
        f["interaction"]["confidence"] = .2
        f["relations"][0]["confidence"] = .2
    records = structured_records("ep_1", uncertain)
    assert all(not r.evolution and not r.action and not r.space.relations for r in records)
    backwards = [frame(0, phase="release"), frame(1, "06:01", "grasp")]
    processes = [r for r in structured_records("ep_1", backwards) if r.evolution]
    assert len(processes) == 2
    assert all(len(r.evolution.stages) == 1 for r in processes)  # never merge backwards phases


@pytest.mark.parametrize("description", [
    "The robot is visible in front of a chair holding a ball.",
    "The robot has moved closer to the camera, obscuring the scene.",
    "The robot gripper is open above an object resting on a chair.",
])
def test_furniture_support_and_agent_motion_do_not_require_manipulation(description):
    raw = frame(description=description, interaction=None, relations=[])
    assert VisualFrame.model_validate(raw).interaction is None


def test_different_model_event_ids_link_with_real_source_evidence_only():
    static=frame(0,description="The cup rests on the desk.",interaction=None,
        relations=[{"source":"cup_1","predicate":"on","target":"desk_1","confidence":.9}])
    grasp=frame(1,"06:01")
    grasp["interaction"].update(id="different_grasp",source_id=None)
    released=frame(2,"06:02","release")
    released["interaction"]["id"]="different_release"
    raw=[static,grasp,released]
    before=json.dumps(raw,sort_keys=True)
    records=structured_records("ep_1",raw)
    process=next(r for r in records if r.evolution)
    assert process.evolution.stages[0].state["source_id"]=="desk_1"
    assert "ep_1_image_0" in process.evolution.stages[0].observation_ids
    assert json.loads(process.answer)["raw_interaction_ids"]==["different_grasp","different_release"]
    assert json.dumps(raw,sort_keys=True)==before  # no edits to perception


def test_resume_rejects_evidence_or_configuration_changes(directory):
    freeze(directory / "config.json", {"embedding_model": "first"})
    with pytest.raises(RuntimeError, match="configuration changed"):
        freeze(directory / "config.json", {"embedding_model": "second"})
    records = structured_records("ep_1", history())
    persist_records(directory, "ep_1", records, "flat")
    records[0].answer = "changed content"
    with pytest.raises(RuntimeError, match="evidence changed"):
        persist_records(directory, "ep_1", records, "flat")


def test_goal_frame_must_be_in_cited_evidence():
    records = structured_records("ep_1", history())
    class Client:
        def generate(self, *args):
            return {"goal_frames": [1], "evidence_ids": ["ep_1_frame_0"]}
    answer = FrameAnswerAgent().answer(Query(question="Find the cup"), records, Client(), "live")
    assert answer.evidence_ids == []
    assert json.loads(answer.text)["goal_frames"] == []


def test_source_goal_uses_confirmed_grasp_and_its_observation():
    records = structured_records("ep_1", history())
    process = next(r for r in records if r.evolution)
    query = history_query("Revisit all the desks you picked objects from yesterday.", "ep_1", records)
    query.recall_plan = RecallPlan(goal="source", location_hint="desk", all_targets=True,
                                   previous_day=True)
    answer = planned_goal_answer(query, records)
    assert json.loads(answer.text)["goal_frames"] == [0]
    assert answer.evidence_ids == [process.id]
    assert hint_score("table", "wooden_desk") == 0
    assert hint_score("table", "round_table") == 1

    uncertain = process.model_copy(deep=True)
    uncertain.evolution.stages[0].state.pop("held_by")
    evidence = json.loads(uncertain.answer)
    evidence["observations"][0]["relations"] = []
    uncertain.answer = json.dumps(evidence)
    assert record_match(uncertain, query.recall_plan) == 0
    assert planned_goal_answer(query, [uncertain]) is None


def test_visible_support_then_first_confirmed_transport_binds_source():
    static = frame(0, description="The cup rests on the desk.", interaction=None,
        relations=[{"source": "cup_1", "predicate": "on", "target": "desk_1", "confidence": .9}])
    carried = frame(1, "06:01", "transport")
    carried["interaction"]["source_id"] = None
    settled = frame(2, "06:02", "settled")
    records = structured_records("ep_1", [static, carried, settled])
    process = next(r for r in records if r.evolution)
    assert process.evolution.stages[0].state["source_id"] == "desk_1"
    query = history_query("Revisit the desk you picked an object from yesterday.", "ep_1", records)
    query.recall_plan = RecallPlan(goal="source", location_hint="desk", previous_day=True)
    assert json.loads(planned_goal_answer(query, records).text)["goal_frames"] == [1]


def test_active_interaction_requires_agent_and_visible_hold():
    missing_actor = frame()
    missing_actor["interaction"]["actor_id"] = None
    with pytest.raises(ValueError, match="declared agent"):
        VisualFrame.model_validate(missing_actor)
    missing_grip = frame()
    missing_grip["relations"] = []
    with pytest.raises(ValueError, match="held_by actor"):
        VisualFrame.model_validate(missing_grip)


def test_labels_and_directed_roles_survive_sqlite_and_reject_background_anchor(directory):
    from memory_agent.recall import graph_plan
    raw = history()
    for f in raw:
        f["entities"][0]["labels"] = [{"text":"ceramic mug", "confidence":.92}]
        f["description"] = "The robot manipulates the cup; a laptop is in the background."
    records = structured_records("ep_1", raw)
    process = next(r for r in records if r.evolution)
    store = persist_records(directory, "ep_1", records, "hypergraph")
    loaded = HypergraphMemory(store.path).get(process.id)
    assert loaded.entities == process.entities and loaded.role_bindings == process.role_bindings
    assert {"entity_identity","role_binding"} <= {e.kind for e in store.graph.edges.values()}
    q = history_query("next", "ep_1", records)
    q.recall_plan = RecallPlan(anchor_hint="laptop", relation="after")
    unresolved = graph_plan(records,q)
    assert "Anchor entity is unresolved" in unresolved["warnings"]
    assert not unresolved["anchor_record_ids"] and not unresolved["target_record_ids"] and not unresolved["complete"]
    q.recall_plan = RecallPlan(entity_hint="ceramic mug")
    assert graph_plan(records,q)["targets"][0]["entity_id"] == "cup_1"


def test_single_frame_confirmed_action_is_a_partial_event():
    records = structured_records("ep_1", [frame()])
    process = next(r for r in records if r.evolution)
    assert len(process.evolution.stages) == 1
    assert not json.loads(process.answer)["coverage"]["destination_grounded"]


def test_object_target_after_interaction_uses_final_supported_location():
    from memory_agent.recall import target_groundings
    records=structured_records("ep_1",history())
    process=next(r for r in records if r.evolution)
    targets=target_groundings(process,"entity")
    assert targets[0]["entity_id"]=="cup_1" and targets[0]["frame_index"]==2


def test_semantic_index_uses_grounded_labels_and_filters_uncertain_aliases():
    from memory_agent.semantic import record_text
    raw=frame()
    raw["entities"][0]["labels"]=[{"text":"ceramic mug","confidence":.9},
                                   {"text":"unknown battery","confidence":.3}]
    record=structured_records("ep_1",[raw])[0]
    text=record_text(record)
    assert "ceramic mug" in text and "unknown battery" not in text


def test_promotion_retains_identity_and_role_evidence(directory):
    from memory_agent.models import now
    process=next(r for r in structured_records("ep_1",history()) if r.evolution)
    process=process.model_copy(update={"scope":"short","session_id":"session",
                                       "valid_until":now()+timedelta(hours=1)})
    store=HypergraphMemory(directory/"promotion.sqlite3")
    store.add(process)
    promoted=store.get(store.promote(process.id,"explicit user confirmation"))
    assert promoted.entities==process.entities and promoted.role_bindings==process.role_bindings
    assert promoted.scope=="long" and promoted.session_id is None


def test_role_bindings_cannot_relabel_furniture_as_actor_or_change_source():
    from memory_agent.models import Record
    process=next(r for r in structured_records("ep_1",history()) if r.evolution)
    wrong=process.model_copy(deep=True)
    next(b for b in wrong.role_bindings if b.role=="actor").entity_id="desk_1"
    with pytest.raises(ValueError,match="Actor role"):
        Record.model_validate(wrong.model_dump())
    wrong=process.model_copy(deep=True)
    next(b for b in wrong.role_bindings if b.role=="source").entity_id="box_1"
    with pytest.raises(ValueError,match="contradicts"):
        Record.model_validate(wrong.model_dump())


def test_projection_preserves_every_role_proof_without_repeating_identity_catalogs():
    from memory_agent.recall import compact_record
    process=next(r for r in structured_records("ep_1",history()) if r.evolution)
    view=compact_record(process)
    assert len(view["roles"])<len(process.role_bindings)
    assert {oid for b in view["roles"] for oid in b["observation_ids"]}=={
        oid for b in process.role_bindings for oid in b.observation_ids}
    assert all("entities" not in f for f in view["evidence"]["observations"])


def test_all_source_targets_deduplicate_repeated_visits_to_same_instance():
    from memory_agent.recall import graph_plan
    raw=[frame(),frame(1,"06:01","settled"),frame(2,"06:02","grasp"),frame(3,"06:03","settled")]
    records=structured_records("ep_1",raw)
    q=history_query("sources","ep_1",records)
    q.recall_plan=RecallPlan(goal="source",all_targets=True)
    result=graph_plan(records,q)
    assert len(result["target_record_ids"])==2 and len(result["targets"])==1


def test_grip_without_departure_does_not_certify_a_completed_pickup():
    raw=[frame(),frame(1,"06:01","grasp")]
    records=structured_records("ep_1",raw)
    q=history_query("sources","ep_1",records)
    q.recall_plan=RecallPlan(goal="source",all_targets=True)
    assert all(record_match(r,q.recall_plan)==0 for r in records)
    complete=structured_records("ep_1",history())
    assert any(record_match(r,q.recall_plan)>0 for r in complete)


def test_missing_source_and_incomplete_order_are_not_silently_rendered():
    raw = history()
    raw[0]["interaction"]["source_id"] = None
    records = structured_records("ep_1", raw)
    q = history_query("sources", "ep_1", records)
    q.recall_plan = RecallPlan(goal="source", all_targets=True)
    process = next(r for r in records if r.evolution)
    assert record_match(process,q.recall_plan) == 0
    assert planned_goal_answer(q,records) is None
    records = structured_records("ep_1",history())
    q.recall_plan = RecallPlan(goal="source", all_targets=True, visit_order=[2,1])
    assert planned_goal_answer(q,records) is None


def test_all_completed_placements_do_not_count_an_unfinished_pickup_as_missing():
    from memory_agent.recall import graph_plan
    raw=[frame(),frame(1,"06:01","settled"),frame(2,"06:02","grasp")]
    records=structured_records("ep_1",raw)
    q=history_query("placements","ep_1",records)
    q.recall_plan=RecallPlan(goal="destination",all_targets=True)
    result=graph_plan(records,q)
    assert result["complete"] and len(result["targets"])==1


def test_untouched_inventory_is_resolved_before_top_k(directory):
    from memory_agent.models import RecallResolution
    raw = history()
    raw[0]["entities"].append({"id":"ball_1","name":"blue ball","kind":"object"})
    records = structured_records("ep_1",raw)
    store = persist_records(directory,"ep_1",records,"hypergraph")
    q = history_query("uninteracted", "ep_1",records)
    q.recall_plan = RecallPlan(goal="uninteracted")
    selected,report=store.search(q,top_k=1)
    assert {t["entity_id"] for t in report["plan_evidence"]["targets"]} == {"ball_1"}
    q.recall_resolution = RecallResolution.model_validate(report["plan_evidence"])
    answer = planned_goal_answer(q,selected)
    assert json.loads(answer.text)["goal_frames"] == [0]
    q.space.namespace = "different_episode"
    assert not store.search(q,top_k=1)[0]


@pytest.mark.parametrize("reject_schema", [False, True])
def test_caption_repairs_invalid_output_and_logs_gateway_format(directory, reject_schema):
    frames_dir = directory / "frames" / "ep_1"
    frames_dir.mkdir(parents=True)
    for n in range(2):
        (frames_dir / f"{n:03d}.jpg").write_bytes(b"mock image bytes")
    good = history()[:2]
    bad = json.loads(json.dumps(good))
    bad[0]["relations"][0]["target"] = "not_declared"
    class Client:
        settings = Settings(llm_key="test-only", llm_model="test-only")
        def __init__(self):
            self.calls = []
            self.payloads = []
            self.responses = 0
        def post(self, url, key, payload, role):
            self.payloads.append(payload.copy())
            if reject_schema and payload["response_format"]["type"] == "json_schema":
                raise ModelError("visual_observation: HTTP 400")
            self.calls.append({"role": role})
            self.responses += 1
            output = bad if self.responses == 1 else good
            return {"choices": [{"message": {"content": json.dumps({"observations": output})}}]}
    client = Client()
    frames = extract_batch(client, directory, "ep_1", [0, 1])
    assert len(frames) == 2 and client.responses == 2
    assert "Rejected output" in client.payloads[-1]["messages"][-1]["content"]
    assert client.calls[-1]["response_format"] == ("json_object" if reject_schema else "json_schema")


@pytest.mark.parametrize("single_succeeds", [True, False])
def test_batch_failure_reextracts_single_frames_without_auto_repair(directory, single_succeeds):
    frames_dir = directory / "frames" / "ep_1"
    frames_dir.mkdir(parents=True)
    for n in range(2):
        (frames_dir / f"{n:03d}.jpg").write_bytes(b"mock image")
    good = history()[:2]
    bad = json.loads(json.dumps(good))
    bad[0]["relations"][0]["target"] = "desk_1"  # held by furniture is invalid
    class Client:
        settings = Settings(llm_key="test-only")
        def __init__(self):
            self.calls = []
            self.requests = []
        def post(self, url, key, payload, role):
            request = json.loads(payload["messages"][1]["content"][0]["text"])
            self.requests.append(request)
            self.calls.append({"role": role})
            indices = request["requested_indices"]
            output = good if len(indices) == 1 and single_succeeds else bad
            return {"choices": [{"message": {"content": json.dumps({
                "observations": [output[i] for i in indices]})}}]}
    client = Client()
    checks = directory / "checks.json"
    if single_succeeds:
        result = extract_batch(client, directory, "ep_1", [0, 1], diagnostic_path=checks)
        assert [f.frame_index for f in result] == [0, 1]
        assert result[0].relations[0].predicate == "held_by"
        assert result[0].relations[0].target == "gripper_1"  # from model, never desk->on heuristic
        assert [r["requested_indices"] for r in client.requests] == [[0, 1]] * 3 + [[0], [1]]
        assert "cup_1" in client.requests[-1]["entity_registry"]
    else:
        with pytest.raises(RuntimeError, match="Visual output failed"):
            extract_batch(client, directory, "ep_1", [0, 1], diagnostic_path=checks)
        assert len(client.requests) == 6  # bounded, no endless retries or acceptance
    assert "target kind=place" in json.loads(checks.read_text())[0]["error"]
    assert any(c.get("status") == "batch_failed_split_into_single_frames" for c in client.calls)


def test_balance_error_is_clear_but_does_not_expose_provider_message():
    secret = "fake-secret-never-print"
    client = ModelClients(Settings(llm_key=secret), httpx.MockTransport(lambda request:
        httpx.Response(403, json={"error": {"code": "insufficient_balance", "message": secret}})))
    try:
        with pytest.raises(ModelError, match="insufficient_balance") as error:
            client.post("https://example.test/chat/completions", secret, {"model": "test"}, "vision")
        assert secret not in str(error.value)
    finally:
        client.close()
