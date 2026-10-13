"""Verify graph-local expansion, equal gates, vector contracts and persistent cache."""
from datetime import datetime, timedelta
import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from memory_agent.agents import MemoryCoordinator
from memory_agent.clients import ModelClients, ModelError
from memory_agent.memory import HypergraphMemory
from memory_agent.models import Answer, Judgment, Query, Record, now
from memory_agent.settings import Settings
from memory_agent.semantic import validate_vectors


@pytest.fixture
def store():
    root = Path(__file__).resolve().parents[1]/"runtime"/"memory_tests"/uuid4().hex
    root.mkdir(parents=True)
    memory = HypergraphMemory(root/"memory.sqlite3")
    yield memory
    for name in ("memory.sqlite3","memory.sqlite3-wal","memory.sqlite3-shm"):
        (root/name).unlink(missing_ok=True)
    root.rmdir()


def record(id="seed",question="alpha",**updates):
    return Record.model_validate(dict(id=id,question=question,key="workflow",answer="verified answer",
        source="synthetic",verified=True,kind="experience",event="incident",
        space={"namespace":"demo","entities":["road"]}) | updates)


def query(**updates):
    return Query.model_validate(dict(question="alpha",event="incident",
        space={"namespace":"demo","entities":["road"]}) | updates)


def test_joint_expansion_needs_two_context_types_in_the_same_episode(store):
    seed,neighbor = record(),record("neighbor","unrelated")
    store.add_many([seed,neighbor])
    flat,flat_report = store.search(query(),strategy="flat")
    graph,report = store.search(query())
    assert [r.id for r in flat] == [seed.id]
    assert {r.id for r in graph} == {seed.id,neighbor.id}
    score = report["scores"][neighbor.id]
    assert score["recall_route"] == "joint_hyperedge_expansion"
    assert score["shared_paths"][0]["shared_types"] == ["event","space"]
    assert flat_report["strategy"] == "flat"


def test_one_shared_type_or_a_shared_task_does_not_expand(store):
    store.add_many([record(),record("neighbor","unrelated",event=None)])
    assert [r.id for r in store.search(query())[0]] == ["seed"]


@pytest.mark.parametrize("strategy",["flat","hypergraph"])
def test_high_vector_similarity_cannot_bypass_space_or_state(store,strategy):
    store.add_many([record(),record("wrong","alpha",required_state={"queue":"shrinking"}),
                    record("other","alpha",space={"namespace":"other","entities":["road"]})])
    supplied = []
    def embed(texts):
        supplied.extend(texts)
        return [[1,0] for _ in texts]
    rows,report = store.search(query(state={"queue":"growing"}),strategy=strategy,
                               embedder=embed,embedding_namespace="test:model")
    assert [r.id for r in rows] == ["seed"]
    assert len(supplied) == 2  # query plus one eligible record
    assert len(report["filtered"]) == 2


def test_task_mismatch_is_never_expanded(store):
    store.add_many([record(),record("other","alpha",key="different")])
    assert [r.id for r in store.search(query(key="workflow"))[0]] == ["seed"]


def test_without_explicit_key_expansion_stays_inside_seed_task_family(store):
    store.add_many([record(),record("different","unrelated",key="other")])
    assert [r.id for r in store.search(query())[0]] == ["seed"]


def test_output_rechecks_graph_bindings_after_judgment(store):
    r = record()
    store.add(r)
    agent = MemoryCoordinator(Settings(store_path=store.path))
    def judge(*args,**kwargs):
        with store.connection(write=True) as db:
            db.execute("DELETE FROM hyperedge_members WHERE edge_id=? AND node_id IN "
                       "(SELECT id FROM nodes WHERE kind='space')",(r.id,))
        return Judgment(action="reuse",record_id=r.id,confidence=1,relevance=1,support=2,model="test")
    agent.experience.judge = judge
    agent.answer_agent.answer = lambda *args:Answer(text=r.answer,evidence_ids=[r.id])
    try:
        assert agent.ask(query()).route == "abstain"
    finally:
        agent.close()


def test_embedding_recalls_paraphrase_without_lexical_overlap_and_survives_restart(store):
    r = record(question="beta")
    store.add(r)
    batches = []
    def embed(texts):
        batches.append(texts)
        return [[1,0] for _ in texts]
    assert store.search(query(),strategy="flat")[0] == []
    assert store.search(query(),embedder=embed,embedding_namespace="test:model")[0][0].id == r.id
    loaded = HypergraphMemory(store.path)
    assert loaded.search(query(),embedder=embed,embedding_namespace="test:model")[0][0].id == r.id
    assert [len(batch) for batch in batches] == [2,1]
    loaded.search(query(),embedder=embed,embedding_namespace="test:different-model")
    assert len(batches[-1]) == 2


def test_cache_content_change_is_not_reused(store):
    r = record()
    store.add(r)
    lengths = []
    def embed(texts):
        lengths.append(len(texts))
        return [[1,0] for _ in texts]
    store.search(query(),embedder=embed,embedding_namespace="test")
    updated = r.model_copy(update={"question":"different content"})
    with store.connection(write=True) as db:
        db.execute("UPDATE records SET payload=? WHERE id=?",(updated.model_dump_json(),r.id))
    store.search(query(),embedder=embed,embedding_namespace="test")
    assert lengths == [2,2]


@pytest.mark.parametrize("vectors",[[[]],[[0,0]],[[float("nan"),1]],[[True,0]],[[1,0],[1]],[],[[10**500,1]]])
def test_invalid_vector_batch(vectors):
    with pytest.raises(ValueError):
        validate_vectors(vectors,1)


def test_embedding_failure_and_dimension_change_fall_back(store):
    store.add(record())
    def bad(texts):
        raise ModelError("provider error")
    rows,report = store.search(query(),embedder=bad,embedding_namespace="test")
    assert rows and report["semantic_backend"] == "tfidf"
    assert report["fallback"] == "embedding_unavailable_or_invalid"
    store.search(query(),embedder=lambda texts:[[1,0] for _ in texts],embedding_namespace="test")
    rows,report = store.search(query(),embedder=lambda texts:[[1,0,0] for _ in texts],embedding_namespace="test")
    assert rows and report["fallback"]


def test_incomplete_episode_cannot_borrow_another_experience_binding(store):
    first,second = record(),record("neighbor","alpha")
    store.add_many([first,second])
    with store.connection(write=True) as db:
        db.execute("DELETE FROM hyperedge_members WHERE edge_id=? AND node_id IN "
                   "(SELECT id FROM nodes WHERE kind='space')",(first.id,))
    assert {r.id for r in store.search(query(),strategy="flat")[0]} == {first.id,second.id}
    rows,report = store.search(query())
    assert [r.id for r in rows] == [second.id]
    assert any(f["record_id"] == first.id and f["reason"] == "hyperedge_binding_incomplete" for f in report["filtered"])


def test_trajectory_requires_its_own_entry_and_directed_transitions(store):
    t = datetime.fromisoformat("2026-10-01T09:00:00+08:00")
    r = record(evolution={"namespace":"demo","case_id":"a","stages":[
        {"id":"a1","at":t,"phase":"growth","state":{"queue":"growing"}},
        {"id":"a2","at":t+timedelta(minutes=10),"phase":"recovery"}]})
    store.add(r)
    q = query(at=t+timedelta(days=1),phase="growth",state={"queue":"growing"})
    rows,report = store.search(q)
    assert rows and report["scores"][r.id]["binding"]["trajectory_bound"]
    with store.connection(write=True) as db:
        db.execute("DELETE FROM hyperedges WHERE kind='state_transition'")
    assert store.search(q)[0] == []
    assert store.search(q,strategy="flat")[0]
    assert not store.search(q.model_copy(update={"phase":"recovery"}),strategy="flat")[0]


def test_expired_short_vector_cache_is_pruned(store):
    r = record(scope="short",session_id="alice",valid_until=now()+timedelta(minutes=1))
    store.add(r)
    store.search(query(session_id="alice"),embedder=lambda texts:[[1,0] for _ in texts],embedding_namespace="test")
    with store.connection(write=True) as db:
        assert db.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0] == 1
        db.execute("UPDATE records SET valid_until=? WHERE id=?",((now()-timedelta(seconds=1)).timestamp(),r.id))
    store.prune()
    with store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0] == 0


def test_embedding_http_index_order_and_invalid_indices():
    settings = Settings(embedding_model="test",llm_key="dummy",retries=0)
    def response(request):
        return httpx.Response(200,json={"data":[{"index":1,"embedding":[0,2]},
                                                {"index":0,"embedding":[3,0]}]})
    clients = ModelClients(settings,httpx.MockTransport(response))
    try:
        assert clients.embed(["first","second"]) == [[1,0],[0,1]]
        clients.http.close()
        clients.http = httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(200,
            json={"data":[{"index":0,"embedding":[1,0]},{"index":0,"embedding":[1,0]}]})))
        with pytest.raises(ModelError):
            clients.embed(["first","second"])
    finally:
        clients.close()


def test_mock_ignores_configured_embedding_provider(store):
    settings = Settings(store_path=store.path,embedding_model="configured")
    clients = ModelClients(settings,httpx.MockTransport(lambda request:pytest.fail("mock must stay offline")))
    agent = MemoryCoordinator(settings,clients)
    store.add(record())
    try:
        result = agent.ask(query())
        assert result.route == "reuse" and result.retrieval["semantic_backend"] == "tfidf"
    finally:
        agent.close()


def test_schema_four_upgrade_keeps_record_payload(store):
    r = record()
    store.add(r)
    with store.connection(write=True) as db:
        original = db.execute("SELECT payload FROM records WHERE id=?",(r.id,)).fetchone()[0]
        db.execute("DROP TABLE embedding_cache")
        db.execute("PRAGMA user_version=4")
    loaded = HypergraphMemory(store.path)
    with loaded.connection() as db:
        assert db.execute("SELECT payload FROM records WHERE id=?",(r.id,)).fetchone()[0] == original
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    assert loaded.counts()["schema_version"] == 6


def test_historical_observation_requires_its_own_source_time_and_space_binding(store):
    observation = dict(id="history",source="gantry",modality="series",namespace="demo",entity="road",
        metric="flow",value=100,unit="vehicles/min",observed_at=now()-timedelta(days=1),provenance="synthetic")
    store.add(record(observations=[observation]))
    assert store.search(query())[0]
    with store.connection(write=True) as db:
        db.execute("DELETE FROM hyperedge_members WHERE edge_id IN "
                   "(SELECT id FROM hyperedges WHERE kind='observation') AND node_id IN "
                   "(SELECT id FROM nodes WHERE kind='source')")
    assert not store.search(query())[0]
    assert store.search(query(),strategy="flat")[0]


@pytest.mark.parametrize("strategy",["flat","hypergraph"])
def test_current_conflicting_evidence_blocks_both_strategies(store,strategy):
    store.add(record(evidence_requirements=[{"metric":"flow","unit":"vehicles/min","min_sources":2}]))
    t = now()
    obs = [dict(id=source,source=source,modality="series",namespace="demo",entity="road",
                metric="flow",unit="vehicles/min",value=value,observed_at=t,provenance="synthetic")
           for source,value in [("gantry",100),("counter",200)]]
    called = []
    rows,report = store.search(query(at=t,observations=obs),strategy=strategy,
                               embedder=lambda texts:called.append(texts),embedding_namespace="test")
    assert not rows and not called
    assert report["filtered"][0]["reason"] == "current_evidence_conflict"


def test_live_embedding_and_single_jev_receive_binding_report(store):
    r = record()
    store.add(r)
    seen = []
    settings = Settings(mode="live",store_path=store.path,llm_key="dummy",jev_key="dummy",
                        embedding_model="test-embedding",retries=0)
    def handler(request):
        payload = json.loads(request.content)
        seen.append(request.url.path)
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200,json={"choices":[{"message":{"content":'{"search_text":"alpha"}'}}]})
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200,json={"data":[{"index":i,"embedding":[1,0]} for i in range(len(payload["input"]))]})
        assert payload["state"]["context"]["candidate_bindings"][r.id]["binding"]["episode_bound"]
        return httpx.Response(200,json={"model":"test-jev","answers":{
            "route":{"type":"choice","choice":r.id,"confidence":.98,
                     "probabilities":{r.id:.98,"synthesize":.01,"abstain":.01}},
            "relevant_"+r.id:{"type":"noul","noul":.99},
            "support_"+r.id:{"type":"score","score":2}}})
    agent = MemoryCoordinator(settings,ModelClients(settings,httpx.MockTransport(handler)))
    try:
        result = agent.ask(query())
        assert result.route == "reuse" and result.retrieval["semantic_backend"] == "embedding"
        assert sum(path.endswith("/v1/systemone") for path in seen) == 1
    finally:
        agent.close()
