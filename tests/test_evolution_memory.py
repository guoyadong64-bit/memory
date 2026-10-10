import json
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
import httpx
from pydantic import ValidationError

from memory_agent.agents import MemoryCoordinator
from memory_agent.clients import ModelClients
from memory_agent.memory import HypergraphMemory
from memory_agent.models import Observation, Query, Record, now
from memory_agent.settings import Settings
from memory_agent.observations import ObservationMapping, normalize_rows


def at(day=12,minute=0):
    return datetime.fromisoformat(f"2026-10-{day:02d}T09:{minute:02d}:00+08:00")


def observation(id="current-1",source="gantry",metric="flow",value=100,unit="vehicles/min",stamp=None,**extra):
    return dict(id=id,source=source,modality="time_series",namespace="network",entity="road-1",
                metric=metric,value=value,unit=unit,observed_at=stamp or at(),provenance="synthetic:test",**extra)


def record(**extra):
    data = dict(key="analysis",question="Which analysis workflow?",answer="Check upstream observations",
        verified=True,kind="experience",source="synthetic fixture",
        space={"namespace":"network","entities":["road-1"]},event="incident",
        required_state={"capacity":"reduced"},
        evidence_requirements=[dict(metric="flow",unit="vehicles/min",minimum=80,maximum=120,min_sources=2)],
        observations=[observation(id="historic",stamp=at(11),value=100)],
        evolution={"namespace":"network","case_id":"case-1","stages":[
            {"id":"s1","phase":"growth","at":at(11),"state":{"queue":"growing"},
             "action":"check upstream","observation_ids":["historic"]},
            {"id":"s2","phase":"recovery","at":at(11,10),"state":{"queue":"shrinking"},
             "outcome":"workflow completed"}]})
    return Record.model_validate(data | extra)


def query(**extra):
    data = dict(key="analysis",question="Which analysis workflow?",at=at(),event="incident",
        space={"namespace":"network","entities":["road-1"]},phase="growth",
        state={"queue":"growing","capacity":"reduced"},observations=[
            observation(),observation(id="current-2",source="counter")])
    return Query.model_validate(data | extra)


@pytest.fixture
def store():
    path = Path(__file__).resolve().parents[1] / "runtime" / "memory_tests" / uuid4().hex
    path.mkdir(parents=True)
    memory = HypergraphMemory(path / "memory.sqlite3")
    yield memory
    for name in ("memory.sqlite3","memory.sqlite3-wal","memory.sqlite3-shm"):
        (path / name).unlink(missing_ok=True)
    path.rmdir()


def test_joint_retrieval_graph_and_reload(store):
    r = record()
    store.add(r)
    loaded = HypergraphMemory(store.path)
    assert loaded.retrieve(query())[0].id == r.id
    assert loaded.explain(query())[0]["evidence"]["matched_observations"] == ["current-1","current-2"]
    assert loaded.get(r.id).evolution == r.evolution
    graph = loaded.graph
    assert {"observation","stage_evidence","state_transition","process"} <= {e.kind for e in graph.edges.values()}
    transition = next(n for n in graph.nodes.values() if n.kind == "transition")
    roles = json.loads(transition.value)
    assert graph.nodes[roles["from"]].kind == "stage"
    assert graph.nodes[roles["to"]].kind == "stage"
    assert roles["from"] != roles["to"] and roles["relation"] == "precedes"
    assert loaded.trace("network","case-1")[0]["record_id"] == r.id
    assert loaded.trace("other","case-1") == []


@pytest.mark.parametrize("changes,reason",[
    ({"phase":"recovery"},"phase_mismatch_or_missing"),
    ({"phase":None},"phase_mismatch_or_missing"),
    ({"state":{"queue":"shrinking","capacity":"reduced"}},"state_mismatch_or_missing"),
    ({"at":at(11)},"historical_outcome_not_yet_available"),
    ({"observations":[]},"current_evidence_missing_or_stale"),
    ({"observations":[observation(),observation(id="other",source="counter",value=150)]},"current_evidence_conflict"),
    ({"observations":[observation(),observation(id="other",source="gantry")]},"insufficient_aligned_sources"),
    ({"observations":[observation(value=150),observation(id="other",source="counter",value=150)]},"observation_value_mismatch"),
    ({"observations":[observation(stamp=at(11))]},"current_evidence_missing_or_stale"),
    ({"observations":[observation(stamp=at(12,1))]},"current_evidence_missing_or_stale"),
    ({"observations":[observation(unit="vehicles/hour")]},"current_evidence_missing_or_stale"),
    ({"observations":[observation(quality=.1)]},"current_evidence_missing_or_stale"),
])
def test_reject_unfit_experience(store,changes,reason):
    store.add(record())
    q = query(**changes)
    assert not store.retrieve(q)
    assert store.explain(q)[0]["reason"] == reason


def test_time_alignment_between_sources_and_metrics(store):
    store.add(record())
    q = query(observations=[observation(),observation(id="other",source="counter",stamp=at()-timedelta(seconds=90))])
    assert store.explain(q)[0]["reason"] == "insufficient_aligned_sources"
    other = record(id="multi_metric",evidence_requirements=[
        dict(metric="flow",unit="vehicles/min",min_sources=1),
        dict(metric="event_type",unit="category",equals="incident",min_sources=1)])
    store.add(other)
    q = query(observations=[observation(),observation(id="event",source="event_feed",metric="event_type",
        unit="category",value="incident",stamp=at()-timedelta(seconds=90))])
    explanation = {r["record_id"]:r["reason"] for r in store.explain(q)}
    assert explanation[other.id] == "cross_metric_time_misalignment"


def test_conflict_tolerance_and_namespace_isolation(store):
    store.add(record(evidence_requirements=[dict(metric="flow",unit="vehicles/min",min_sources=2,conflict_tolerance=5)]))
    assert store.retrieve(query(observations=[observation(),observation(id="other",source="counter",value=103)]))
    observations = [observation(),observation(id="other",source="counter")]
    observations[1]["namespace"] = "other"
    assert not store.retrieve(query(observations=observations))


def test_chronology_reference_validation():
    original = record().model_dump(mode="json")
    original["evolution"]["stages"][1]["at"] = original["evolution"]["stages"][0]["at"]
    with pytest.raises(ValidationError):
        Record.model_validate(original)
    original = record().model_dump(mode="json")
    original["evolution"]["stages"][0]["observation_ids"] = ["unknown"]
    with pytest.raises(ValidationError):
        Record.model_validate(original)
    original = record().model_dump(mode="json")
    original["observations"][0]["observed_at"] = at(11,1).isoformat()
    with pytest.raises(ValidationError):
        Record.model_validate(original)
    with pytest.raises(ValidationError):
        Observation.model_validate(observation(value=float("nan")))


def test_trace_session_isolation_promotion_and_cleanup(store):
    r = record(scope="short",session_id="alice",valid_until=now()+timedelta(minutes=10))
    store.add(r)
    assert not store.trace("network","case-1")
    assert not store.trace("network","case-1","bob")
    assert store.trace("network","case-1","alice")
    promoted = store.get(store.promote(r.id,"explicit confirmation"))
    assert promoted.evolution == r.evolution and promoted.evidence_requirements == r.evidence_requirements
    expired = record(scope="short",session_id="bob",valid_until=now()-timedelta(seconds=1))
    store.add(expired)
    store.prune()
    assert expired.id not in store.graph.records
    assert store.retrieve(query())[0].id == promoted.id
    with store.connection() as db:
        assert not db.execute("PRAGMA foreign_key_check").fetchall()


def test_end_to_end_mock_stage_change_abstains(store):
    agent = MemoryCoordinator(Settings(mode="mock",store_path=store.path))
    try:
        agent.memory.add(record())
        assert agent.ask(query()).route == "reuse"
        assert agent.ask(query(phase="recovery")).route == "abstain"
        assert agent.ask(query(observations=[])).route == "abstain"
    finally:
        agent.close()


def mapping(**extra):
    return ObservationMapping.model_validate(dict(source="gantry",modality="time_series",namespace="network",
        metric="flow",input_unit="vehicles/hour",output_unit="vehicles/min",scale=1/60,
        entity_aliases={"G001":"road-1"},fields={"id":"row_id","entity":"gate_id","value":"count",
        "observed_at":"time","provenance":"uri","unit":"unit"}) | extra)


def raw(**extra):
    return dict(row_id="r1",gate_id="G001",count=6000,time=at().isoformat(),uri="synthetic:raw",unit="vehicles/hour") | extra


def test_normalize_alias_units_and_provenance():
    o = normalize_rows([raw()],mapping())[0]
    assert o.entity == "road-1" and o.value == 100 and o.unit == "vehicles/min"
    assert o.source == "gantry" and o.id == "gantry:r1" and o.provenance == "synthetic:raw"


@pytest.mark.parametrize("rows", [[raw(unit="unknown")],[raw(count="6000")],
    [raw(count=True)],[raw(),raw()], [{"row_id":"missing"}], [raw(time="2026-10-12T09:00:00")]])
def test_normalization_rejects_ambiguous_data(rows):
    with pytest.raises((ValueError,ValidationError)):
        normalize_rows(rows,mapping())


def test_source_count_cannot_mix_entities(store):
    store.add(record())
    q = query().model_dump(mode="json")
    q["space"]["entities"] = ["road-1","road-2"]
    q["observations"][1]["entity"] = "road-2"
    assert not store.retrieve(Query.model_validate(q))


def test_same_source_same_timestamp_cannot_report_two_values():
    with pytest.raises(ValidationError):
        query(observations=[observation(),observation(id="duplicate-time",value=200)])


def test_prune_rebuilds_shared_process_members(store):
    active = record(scope="short",session_id="alice",valid_until=now()+timedelta(minutes=10))
    expired_data = active.model_dump(mode="json")
    expired_data.update(id="expired",valid_until=(now()-timedelta(seconds=1)).isoformat())
    expired_data["evolution"]["stages"][1]["at"] = at(11,30).isoformat()
    expired = Record.model_validate(expired_data)
    store.add_many([active,expired])
    assert len([n for n in store.graph.nodes.values() if n.kind == "stage"]) == 3
    store.prune()
    graph = store.graph
    assert len([n for n in graph.nodes.values() if n.kind == "stage"]) == 2
    process = next(e for e in graph.edges.values() if e.kind == "process")
    assert len(process.members) == 3


def test_live_contract_sends_current_and_historical_process_once(store):
    r = record()
    store.add(r)
    settings = Settings(mode="live",llm_key="test",jev_key="test",store_path=store.path,retries=0)
    requested = []
    def handler(request):
        requested.append(request.url.path)
        payload = json.loads(request.content)
        if request.url.path.endswith("chat/completions"):
            return httpx.Response(200,json={"choices":[{"message":{"content":'{"search_text":"analysis workflow"}'}}]})
        assert payload["state"]["query"]["phase"] == "growth"
        assert len(payload["state"]["query"]["observations"]) == 2
        assert payload["state"]["records"][0]["evolution"]["stages"][1]["phase"] == "recovery"
        assert payload["state"]["records"][0]["observations"][0]["id"] == "historic"
        return httpx.Response(200,json={"model":"test-jev","answers":{
            "route":{"type":"choice","choice":r.id,"confidence":1,
                     "probabilities":{r.id:1,"synthesize":0,"abstain":0}},
            "relevant_"+r.id:{"type":"noul","noul":1},
            "support_"+r.id:{"type":"score","score":2}}})
    agent = MemoryCoordinator(settings,ModelClients(settings,httpx.MockTransport(handler)))
    try:
        result = agent.ask(query())
        assert result.route == "reuse" and result.judgment.support == 2
        assert sum(path.endswith("systemone") for path in requested) == 1
    finally:
        agent.close()
