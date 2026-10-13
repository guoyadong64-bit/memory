from datetime import datetime, timedelta, timezone
import json

from memory_agent.models import Query, Record, RecallPlan, Evolution, Stage, SpatialContext, TemporalContext
from memory_agent.recall import parse_plan, graph_plan, record_match, compact_record
from memory_agent.retrieval import context_coverage


def episode(identity, start, end, namespace="room"):
    at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return Record(id=identity, key="history", question=identity, answer=json.dumps({"entities": {identity:identity}}),
        source="synthetic", verified=True, event="visual_history",
        time=TemporalContext(event_at=at+timedelta(minutes=end)),
        space=SpatialContext(namespace=namespace, entities=[namespace,identity]),
        evolution=Evolution(namespace=namespace,case_id=identity,stages=[
            Stage(id="start",phase="grasp",at=at+timedelta(minutes=start),state={"object_id":identity}),
            Stage(id="end",phase="release",at=at+timedelta(minutes=end),state={"object_id":identity})]))


def test_task_parser_extracts_operators_not_answers():
    p=parse_plan("Navigate to the first receptacle that you picked an object from.")
    assert p.goal=="source" and p.ordinal==1
    p=parse_plan("Navigate to the object you interacted with 2 interactions after ending the interaction with brass key.")
    assert p.relation=="after" and p.offset==2 and p.anchor_hint=="brass key"
    p=parse_plan("Revisit all the receptacles you placed objects on yesterday.")
    assert p.goal=="destination" and p.all_targets and p.previous_day
    p=parse_plan("Revisit all the tables you picked objects from. The order to revisit them is: third, first, second.")
    assert p.visit_order==[3,1,2] and p.location_hint=="tables"
    assert parse_plan("Navigate to the farthest receptacle.").requires_geometry


def test_relative_operator_uses_event_end_and_offset_not_high_semantic_similarity():
    rows=[episode("key",0,2),episode("coin",1,3),episode("cup",4,5),episode("parcel",7,8)]
    q=Query(question="next",purpose="recall",recall_plan=RecallPlan(anchor_hint="key",relation="after",offset=2))
    plan=graph_plan(rows,q)
    assert plan["record_ids"]==["key","parcel"]  # overlapping coin is not after key's end
    q.recall_plan.anchor_hint="unseen"
    unresolved=graph_plan(rows,q)
    assert unresolved["record_ids"] and not unresolved["target_record_ids"] and not unresolved["complete"]


def test_namespace_and_ambiguity_are_not_invented_away():
    q=Query(question="next",purpose="recall",recall_plan=RecallPlan(anchor_hint="cup",relation="after"))
    rows=[episode("cup",0,1),episode("cup_2",3,4),episode("parcel",6,7)]
    # Two exact/partial aliases are not equally strong unless names explicitly coincide.
    for r in rows[:2]: r.answer=json.dumps({"entities": {r.id:"cup"}})
    assert "Multiple anchor" in graph_plan(rows,q)["warnings"][0]
    q.space=SpatialContext(namespace="another_room",entities=["another_room"])
    assert graph_plan(rows,q)["record_ids"]==[]


def test_context_score_measures_specific_object_not_field_presence():
    def fact(identity):
        r=episode(identity,0,1)
        r.evolution=None
        r.answer=json.dumps({"frame_index":0,"entities":[{"id":identity,"name":identity,"kind":"object"}],"interaction":None})
        return r
    q=Query(question="cup",purpose="recall",recall_plan=RecallPlan(entity_hint="cup"))
    assert context_coverage(fact("cup"),q)==1
    assert context_coverage(fact("ball"),q)==0
    assert "observations" not in compact_record(fact("cup"))


def test_intended_destination_does_not_certify_a_completed_placement():
    r=episode("cup",0,1)
    r.evolution.stages[0].state["target_id"]="table"
    r.evolution.stages[1].phase="transport"
    r.evolution.stages[1].state["target_id"]="table"
    assert record_match(r,RecallPlan(goal="destination"))==0
