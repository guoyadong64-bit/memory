"""Regression cases for identity attribution, partial facts and evidence budgets."""
import json

import pytest

from benchmarks.findingdory_memory import Entity, history_query, structured_records
from memory_agent.identity import contextual_labels, direct_lift
from memory_agent.models import RecallPlan
from memory_agent.recall import compact_record, graph_plan, pack_evidence, record_match
from test_findingdory_structured import frame, history


def test_context_name_enrichment_does_not_copy_background_or_merge_entities():
    entities = [Entity(id="device_1",name="handheld device",kind="object",
                       labels=[{"text":"handheld device","confidence":.9}]),
                Entity(id="cup_1",name="white ceramic cup",kind="object",
                       labels=[{"text":"ceramic cup","confidence":.9}])]
    result = contextual_labels(entities,"A white handheld music device lies next to a white ceramic cup.")
    assert ("device_1","white handheld music device",.9) in result
    assert not any(identity == "cup_1" and "music" in text for identity,text,_ in result)
    # Two indistinguishable device descriptions cannot establish a unique instance.
    entities.append(entities[0].model_copy(update={"id":"device_2"}))
    assert not any("music" in text for _,text,_ in contextual_labels(entities,
        "A white handheld music device is visible."))


def test_enriched_label_keeps_observation_provenance_and_not_a_new_instance():
    raw = frame(description="A white ceramic travel cup is visible in the gripper.")
    raw["entities"][0]["name"] = "white ceramic cup"
    raw["entities"][0]["labels"] = [{"text":"ceramic cup","confidence":.9}]
    records = structured_records("sample",[raw])
    entity = next(e for e in records[0].entities if e.id == "cup_1")
    label = next(l for l in entity.labels if "travel" in l.text)
    assert label.observation_ids == ["sample_image_0"] and label.confidence == .9
    assert len({e.id for e in records[0].entities if e.id == "cup_1"}) == 1


def test_single_frame_lift_supports_source_without_certifying_destination():
    raw = frame(description="The robot gripper is actively lifting a white ceramic cup from the wooden desk.")
    raw["entities"][0]["labels"] = [{"text":"ceramic cup","confidence":.9}]
    records = structured_records("sample",[raw])
    process = next(r for r in records if r.evolution)
    coverage = json.loads(process.answer)["coverage"]
    assert coverage["local_pickup_confirmed"] and not coverage["event_complete"]
    assert record_match(process,RecallPlan(goal="source")) == 1
    assert record_match(process,RecallPlan(goal="destination")) == 0
    query = history_query("source", "sample",records)
    query.recall_plan = RecallPlan(goal="source")
    assert graph_plan(records,query)["targets"][0]["frame_index"] == 0


def test_adjacent_held_object_on_support_recovers_landing_with_provenance():
    from memory_agent.recall import compact_record
    held = frame(0, "06:00", "transport",
                 description="The robot gripper is holding a cup above the desk.")
    supported = frame(1, "06:01", "settled", interaction=None,
                      description="The robot gripper is open and the cup rests on the box.",
                      relations=[{"source":"cup_1", "predicate":"on",
                                  "target":"box_1", "confidence":.9}])
    records = structured_records("sample", [held, supported])
    process = next(r for r in records if r.evolution)
    assert json.loads(records[1].answer)["interaction"] is None
    assert process.evolution.stages[-1].phase == "settled"
    assert not json.loads(process.answer)["coverage"]["event_complete"]
    assert json.loads(process.answer)["inferred_bindings"][0]["direct_action_observed"] is False
    assert next(b for b in process.role_bindings if b.role == "destination").basis == "linked"
    assert next(r for r in compact_record(process)["roles"] if r["role"] == "destination")["basis"] == "linked"
    query = history_query("Revisit all the receptacles you placed objects on yesterday.", "sample", records)
    query.recall_plan = RecallPlan(goal="destination", all_targets=True)
    assert [(t["frame_index"],t["entity_id"]) for t in graph_plan(records,query)["targets"]] == [(1,"box_1")]


def test_static_support_or_nonadjacent_frames_do_not_create_a_placement():
    held = frame(0, "06:00", "transport")
    supported = frame(1, "06:01", "settled", interaction=None,
                      description="The cup rests on the box while the robot is elsewhere.",
                      entities=[e for e in frame()["entities"] if e["id"] != "gripper_1"],
                      relations=[{"source":"cup_1", "predicate":"on",
                                  "target":"box_1", "confidence":.9}])
    assert all(s.phase != "settled" for r in structured_records("sample",[held,supported])
               if r.evolution for s in r.evolution.stages)
    supported["frame_index"] = 2
    supported["entities"].append(frame()["entities"][1])
    supported["description"] = "The robot gripper is open and the cup rests on the box."
    assert all(s.phase != "settled" for r in structured_records("sample",[held,supported])
               if r.evolution for s in r.evolution.stages)


@pytest.mark.parametrize("description",[
    "The robot gripper is holding a white cup above the wooden desk.",
    "The robot gripper is not lifting a white cup from the wooden desk.",
    "The robot gripper may be lifting a white cup from the wooden desk.",
    "The robot gripper attempts lifting a white cup from the wooden desk.",
    "A person is lifting a white cup; the robot gripper is nearby.",
    "The robot gripper is lifting a blue ball while holding the white cup."])
def test_contact_background_negation_and_other_object_do_not_certify_lift(description):
    assert not direct_lift(frame(description=description))


def test_budget_skips_oversized_record_and_marks_missing_required_proof():
    evidence = [{"id":"large","content":"x"*5000,"frame_indices":[0]},
                {"id":"small","content":"supported fact","frame_indices":[1]}]
    resolution = {"targets":[{"record_id":"large","entity_id":"a","role":"source","frame_index":0},
                              {"record_id":"small","entity_id":"b","role":"source","frame_index":1}],
                  "anchor_record_ids":[],"complete":True,"warnings":[]}
    kept, revised = pack_evidence(evidence,resolution,max_bytes=500)
    assert [item["id"] for item in kept] == ["small"]
    assert not revised["complete"] and len(revised["targets"]) == 1
    assert "omitted" in revised["warnings"][0]
    _, alternative = pack_evidence(evidence,resolution,max_bytes=500,require_all_targets=False)
    assert alternative["complete"]  # any-one-target does not require all alternative candidates


def test_target_projection_retains_frame_and_observation_ids():
    records = structured_records("sample",history())
    query = history_query("source", "sample",records)
    query.recall_plan = RecallPlan(goal="source")
    resolution = graph_plan(records,query)
    process = next(r for r in records if r.evolution)
    evidence = [{"id":process.id,"content":compact_record(process),"frame_indices":[0,1,2]}]
    kept, revised = pack_evidence(evidence,resolution)
    view = kept[0]
    assert revised["complete"] and 0 in view["frame_indices"]
    assert any("sample_image_0" in f["observation_ids"] for f in view["content"]["evidence"]["observations"])
    assert set(view["frame_indices"]) <= {0,1,2}


def test_selected_event_without_requested_role_is_not_complete():
    records = structured_records("sample",[frame()])
    query = history_query("source", "sample",records)
    query.recall_plan = RecallPlan(goal="source",ordinal=1)
    resolution = graph_plan(records,query)
    assert not resolution["targets"] and not resolution["complete"]


def test_untouched_query_prefers_clear_item_over_partial_fixture_and_attempted_item():
    raw = history()
    for f in raw:
        f["entities"].extend([
            {"id":"shelf_partial","name":"shelf fragment","kind":"object",
             "attributes":{"visibility":"partial","role":"fixture"}},
            {"id":"bead_1","name":"blue bead","kind":"object",
             "labels":[{"text":"bead","confidence":.9}],
             "attributes":{"visibility":"clear","role":"movable"}}])
        f["relations"].append({"source":"bead_1","predicate":"on","target":"desk_1","confidence":.9})
    records = structured_records("sample",raw)
    query = history_query("untouched","sample",records)
    query.recall_plan = RecallPlan(goal="uninteracted")
    targets = graph_plan(records,query)["targets"]
    assert targets[0]["entity_id"] == "bead_1"
    assert "shelf_partial" not in {t["entity_id"] for t in targets}


def test_sequence_extractor_sends_before_and_after_context_without_questions(tmp_path):
    from benchmarks.findingdory_structured import extract_batch
    from memory_agent.settings import Settings
    from benchmarks.findingdory_sequence import SEQUENCE_SYSTEM
    directory = tmp_path/"frames"/"sample"
    directory.mkdir(parents=True)
    for index in range(6):
        (directory/f"{index:03d}.jpg").write_bytes(b"test image")
    class Client:
        settings = Settings(llm_key="test",llm_model="qwen-test")
        calls = []
        request = None
        def post(self,url,key,payload,role):
            self.request = payload
            self.calls.append({"role":role})
            return {"choices":[{"message":{"content":json.dumps({"observations":[frame(2),frame(3)]})}}]}
    client = Client()
    extract_batch(client,tmp_path,"sample",[2,3],system=SEQUENCE_SYSTEM,context_before=2,context_after=2)
    texts = [part["text"] for part in client.request["messages"][1]["content"] if part["type"] == "text"]
    assert any("frame_index=0; context_only=True" in text for text in texts)
    assert any("frame_index=5; context_only=True" in text for text in texts)
    request = json.loads(texts[0])
    assert not {"question","answer","gold"} & request.keys()


def test_open_carry_survives_sparse_frames_and_changed_model_event_id():
    carried = frame(10,"06:10","transport")
    carried["interaction"]["id"] = "model_renamed_event"
    released = frame(11,"06:11","release")
    released["interaction"]["id"] = "another_model_event_id"
    records = structured_records("sample",[frame(),carried,released])
    processes = [r for r in records if r.evolution]
    assert len(processes) == 1
    assert processes[0].evolution.stages[1].state["source_id"] == "desk_1"
    assert json.loads(processes[0].answer)["coverage"]["unobserved_intervals"] == [[0,10]]
    # A new grasp is not proof of uninterrupted transport across the gap.
    repeat = frame(10,"06:10","grasp")
    repeat["interaction"]["id"] = "new_pickup"
    assert len([r for r in structured_records("sample",[frame(),repeat]) if r.evolution]) == 2


def test_visual_answer_images_cannot_escape_retrieved_frames():
    from benchmarks.findingdory_visual_answer import select_images
    observations = [{"frame_index":i,"interaction":{"id":f"event_{i//4}"}} for i in range(40)]
    evidence = [{"id":"retrieved","frame_indices":[0,1,7,13,19,25,30,31]}]
    chosen = select_images(evidence,observations,limit=4)
    assert len(chosen) <= 4 and set(chosen) <= set(evidence[0]["frame_indices"])
    assert 39 not in chosen


def test_unresolved_inventory_keeps_boundaries_and_inherits_one_namespace():
    records = structured_records("sample",history())
    query = history_query("unknown","sample",records)
    query.recall_plan = RecallPlan(anchor_hint="unobserved",relation="after")
    resolution = graph_plan(records,query)
    process = next(r for r in records if r.evolution)
    evidence = [{"id":process.id,"content":compact_record(process),"frame_indices":[0,1,2]}]
    kept,revised = pack_evidence(evidence,resolution)
    assert set(kept[0]["frame_indices"]) == {0,1,2}
    assert kept[0]["content"]["entity_namespace"] == "sample"
    assert all("namespace" not in entity for entity in kept[0]["content"]["entities"])
    assert not revised["complete"] and not revised["targets"]


def test_category_mismatch_keeps_event_proofs_without_certifying_category():
    records = structured_records("sample",history())
    query = history_query("all places", "sample",records)
    query.recall_plan = RecallPlan(goal="destination",all_targets=True,location_hint="unrecognized_category")
    resolution = graph_plan(records,query)
    process = next(r for r in records if r.evolution)
    assert process.id in resolution["record_ids"]
    assert not resolution["targets"] and not resolution["complete"]
    assert any("category/role" in warning for warning in resolution["warnings"])


def test_candidate_event_reserved_before_large_unrelated_frame():
    records = structured_records("sample",history())
    process = next(r for r in records if r.evolution)
    event = {"id":process.id,"content":compact_record(process),"frame_indices":[0,1,2]}
    unrelated = {"id":"unrelated","content":"x"*5000,"frame_indices":[99]}
    resolution = {"record_ids":[process.id],"targets":[],"anchor_record_ids":[],"complete":False,"warnings":[]}
    event_only,_ = pack_evidence([event],resolution)
    cap = len(json.dumps(event_only,ensure_ascii=False).encode()) + 10
    kept,_ = pack_evidence([unrelated,event],resolution,max_bytes=cap)
    assert [item["id"] for item in kept] == [process.id]
    assert set(kept[0]["frame_indices"]) == {0,1,2}
