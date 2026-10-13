import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from memory_agent.clients import ModelClients
from memory_agent.memory import HypergraphMemory
from memory_agent.models import Query, Record, SpatialContext, TemporalContext, now
from memory_agent.settings import Settings
from memory_agent.spatiotemporal import context_rejection


@pytest.fixture
def store():
    root = Path(__file__).resolve().parents[1] / "runtime" / "memory_tests" / uuid4().hex
    root.mkdir(parents=True)
    memory = HypergraphMemory(root / "memory.sqlite3")
    yield memory
    for name in ("memory.sqlite3", "memory.sqlite3-wal", "memory.sqlite3-shm"):
        (root / name).unlink(missing_ok=True)
    root.rmdir()


def stamp(value="2026-10-12T08:30:00+08:00"):
    return datetime.fromisoformat(value)


def record(**changes):
    values = dict(key="analysis.workflow", question="How to analyze?", answer="Check inputs first",
                  source="synthetic test", verified=True,
                  time=TemporalContext(event_at=stamp("2026-10-11T08:30:00+08:00"),
                      weekdays=[0,1,2,3,4], local_start_minute=480, local_end_minute=600,
                      utc_offset_minutes=480),
                  space=SpatialContext(namespace="demo", entities=["region-A"]),
                  event="congestion", action="check", outcome="passed")
    return Record(**(values | changes))


def query(**changes):
    values = dict(question="How to analyze?", key="analysis.workflow", at=stamp(), event="congestion",
                  space=SpatialContext(namespace="demo", entities=["road-1"], relations=[
                      {"source":"road-1", "predicate":"within", "target":"region-A"}]))
    return Query(**(values | changes))


@pytest.mark.parametrize("change,reason", [
    ({"at":None}, "query_time_missing"),
    ({"at":stamp("2026-10-10T08:30:00+08:00")}, "future_observation"),
    ({"at":stamp("2026-10-17T08:30:00+08:00")}, "weekday_mismatch"),
    ({"at":stamp("2026-10-12T10:00:00+08:00")}, "periodic_window_mismatch"),
    ({"space":None}, "space_mismatch_or_missing"),
    ({"event":"other"}, "event_mismatch_or_missing"),
])
def test_hard_context_gates(store, change, reason):
    r = record()
    store.add(r)
    q = query(**change)
    assert context_rejection(r,q) == reason
    assert store.retrieve(q) == []


def test_direction_namespace_and_transitive_containment(store):
    r = record()
    store.add(r)
    for namespace, relations in [
        ("other", [{"source":"road-1","predicate":"within","target":"region-A"}]),
        ("demo", [{"source":"region-A","predicate":"within","target":"road-1"}]),
        ("demo", [{"source":"road-1","predicate":"adjacent","target":"region-A"}]),
    ]:
        assert not store.retrieve(query(space=SpatialContext(namespace=namespace,entities=["road-1"],relations=relations)))
    q = query(space=SpatialContext(namespace="demo",entities=["road-1"],relations=[
        {"source":"road-1","predicate":"within","target":"city"},
        {"source":"city","predicate":"within","target":"region-A"}]))
    assert store.retrieve(q)[0].id == r.id
    assert store.retrieval_scores(r,q)["spatial"] == .5


def test_persistence_graph_roles_and_schema_upgrade(store):
    r = record(space=SpatialContext(namespace="demo", entities=["region-A"], relations=[
        {"source":"road-1","predicate":"within","target":"region-A"}]))
    store.add(r)
    with store.connection() as db:
        db.execute("PRAGMA user_version=2")
    loaded = HypergraphMemory(store.path)
    assert loaded.retrieve(query())[0].model_dump() == r.model_dump()
    assert loaded.counts()["schema_version"] == 6
    assert loaded.counts()["edge_types"]["spatial_relation"] == 1
    assert loaded.graph.version == 3
    nodes = loaded.graph.nodes
    assert {nodes[n].kind for n in loaded.graph.edges[r.id].members} >= {"time","space","event","action","outcome"}
    relation = next(n for n in nodes.values() if n.kind == "spatial_relation")
    assert json.loads(relation.value) == ["demo","road-1","within","region-A"]
    with loaded.connection() as db:
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_context_specific_facts_do_not_supersede_each_other(store):
    first = record()
    second = record(space=SpatialContext(namespace="demo",entities=["region-B"]))
    store.add_many([first,second])
    assert store.get(first.id).superseded_by is None
    replacement = record(answer="Updated")
    store.add(replacement)
    assert store.get(first.id).superseded_by == replacement.id
    assert store.get(second.id).superseded_by is None


def test_interval_and_storage_expiry_are_distinct(store):
    r = record(time=TemporalContext(applicable_from=stamp(),applicable_until=stamp()+timedelta(hours=1)))
    store.add(r)
    assert store.retrieve(query())
    assert not store.retrieve(query(at=stamp()+timedelta(hours=1)))
    assert store.get(r.id).valid_until is None
    expired = record(valid_until=now()-timedelta(seconds=1))
    store.add(expired)
    assert expired.id not in [r.id for r in store.retrieve(query())]


def test_timezone_and_overnight_window():
    r = record(time=TemporalContext(local_start_minute=1380,local_end_minute=120,utc_offset_minutes=480))
    assert context_rejection(r,query(at=stamp("2026-10-12T16:30:00+00:00"))) is None
    assert context_rejection(r,query(at=stamp("2026-10-12T18:00:00+00:00"))) == "periodic_window_mismatch"


@pytest.mark.parametrize("time", [
    {"event_at":"2026-10-12T08:00:00"}, {"weekdays":[7]},
    {"weekdays":[0,0]}, {"local_start_minute":480},
    {"local_start_minute":480,"local_end_minute":480},
    {"applicable_from":stamp(),"applicable_until":stamp()},
])
def test_invalid_temporal_context(time):
    with pytest.raises(ValidationError):
        TemporalContext(**time)


def test_promotion_preserves_context_and_prune_shared_relations(store):
    space = SpatialContext(namespace="demo",entities=["region-A"],relations=[
        {"source":"road-1","predicate":"within","target":"region-A"}])
    r = record(time=None,space=space,scope="short",session_id="s",valid_until=now()+timedelta(minutes=5))
    store.add(r)
    promoted = store.get(store.promote(r.id,"confirmed"))
    assert promoted.space == r.space and promoted.action == r.action and promoted.outcome == r.outcome
    expired = record(time=None,space=space,scope="short",session_id="other",valid_until=now()-timedelta(seconds=1))
    store.add(expired)
    store.prune()
    assert store.counts()["edge_types"]["spatial_relation"] == 1
    assert store.retrieve(query())[0].id == promoted.id


def test_jev_receives_serialized_context_and_exposes_support():
    r,q = record(),query()
    def handler(request):
        payload = json.loads(request.content)
        assert payload["state"]["query"]["at"] == q.at.isoformat()
        assert payload["state"]["records"][0]["space"]["entities"] == ["region-A"]
        assert "time" in payload["questions"]["route"]["instructions"]
        return httpx.Response(200,json={"model":"test-jev","answers":{
            "route":{"type":"choice","choice":r.id,"confidence":1,
                     "probabilities":{r.id:1,"synthesize":0,"abstain":0}},
            "relevant_"+r.id:{"type":"noul","noul":1},
            "support_"+r.id:{"type":"score","score":2}}})
    clients = ModelClients(Settings(retries=0),httpx.MockTransport(handler))
    try:
        judgment = clients.judge(q,[r])
        assert judgment.action == "reuse" and judgment.support == 2
    finally:
        clients.close()
