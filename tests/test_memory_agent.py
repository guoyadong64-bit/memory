"""Offline lifecycle tests and real HTTP-contract tests using MockTransport."""
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import timedelta
from pathlib import Path
from uuid import uuid4
import sqlite3

import httpx
import pytest

from memory_agent.agents import MemoryCoordinator
from memory_agent.clients import ModelClients, ModelError
from memory_agent.memory import HypergraphMemory
from memory_agent.models import Query, Record, now
from pydantic import ValidationError
from memory_agent.settings import Settings
from memory_agent.probe import probe


@pytest.fixture
def tmp_path():
    # Avoid mode=0o700 temporary directories, inaccessible in this Windows sandbox.
    root = Path(__file__).resolve().parents[1] / "runtime" / "memory_tests"
    path = root / uuid4().hex
    path.mkdir(parents=True)
    yield path
    for name in ("memory.sqlite3", "memory.sqlite3-wal", "memory.sqlite3-shm", "legacy.json", "legacy2.json", ".env"):
        (path / name).unlink(missing_ok=True)
    path.rmdir()


def record(**changes):
    return Record(**dict(key="report.format", question="项目报告用什么格式？",
                         answer="Markdown", source="test evidence", verified=True,
                         conditions={"project": "demo"}, **changes))


def query():
    return Query(question="项目报告用什么格式？", key="report.format",
                 conditions={"project": "demo"})


def test_persistence_and_joint_conditions(tmp_path):
    store = HypergraphMemory(tmp_path / "memory.sqlite3")
    r = record()
    store.add(r)
    loaded = HypergraphMemory(store.path)
    assert loaded.retrieve(query())[0].id == r.id
    assert not loaded.retrieve(Query(question=r.question, key=r.key))
    assert not loaded.retrieve(query().model_copy(update={"conditions": {"project": "other"}}))
    edge = loaded.graph.edges[r.id]
    assert edge.kind == "episode" and len(edge.members) == 5
    assert {loaded.graph.nodes[n].kind for n in edge.members} == {
        "intent", "condition", "answer", "evidence", "feedback"}


def test_verified_update_and_failure_never_resurrect_old_assertion(tmp_path):
    store = HypergraphMemory(tmp_path / "memory.sqlite3")
    old = record()
    store.add(old)
    new = old.model_copy(update={"id": "new", "answer": "PDF", "created_at": now()})
    store.add(new)
    assert [r.id for r in store.retrieve(query())] == [new.id]
    store.fail(new.id, "user corrected the preference")
    loaded = HypergraphMemory(store.path)
    assert not loaded.retrieve(query())
    assert loaded.graph.records[old.id].superseded_by == new.id
    assert any(e.kind == "invalidation" for e in loaded.graph.edges.values())


def test_unverified_expired_and_older_updates(tmp_path):
    store = HypergraphMemory(tmp_path / "memory.sqlite3")
    r = record()
    store.add(r)
    draft = r.model_copy(update={"id": "draft", "verified": False})
    store.add(draft)
    assert [v.id for v in store.retrieve(query())] == [r.id]
    expired = r.model_copy(update={"id": "expired", "key": "old.format",
                                   "valid_until": now() - timedelta(days=1)})
    store.add(expired)
    assert not store.retrieve(query().model_copy(update={"key": "old.format"}))
    older = r.model_copy(update={"id": "older", "created_at": now() - timedelta(days=1)})
    with pytest.raises(ValueError, match="older"):
        store.add(older)
    assert "older" not in store.graph.records


def test_mock_never_calls_http(tmp_path):
    def forbidden(request):
        pytest.fail("Mock mode must not call a model")
    settings = Settings(store_path=tmp_path / "memory.sqlite3")
    clients = ModelClients(settings, transport=httpx.MockTransport(forbidden))
    agent = MemoryCoordinator(settings, clients)
    try:
        assert agent.ask(query()).route == "abstain"
        r = record()
        agent.memory.add(r)
        result = agent.ask(query())
        assert result.route == "reuse"
        assert result.answer.evidence_ids == [r.id]
        assert result.events == []
    finally:
        agent.close()


def jev_reply(rid, confidence=0.98, score=2):
    return {"model": "jev-1.13.0", "usage": {"input_tokens": 12}, "answers": {
        "route": {"type": "choice", "choice": rid, "confidence": confidence,
                  "probabilities": {rid: 0.98, "synthesize": 0.01, "abstain": 0.01}},
        "relevant_" + rid: {"type": "noul", "noul": 0.95},
        "support_" + rid: {"type": "score", "score": score}}}


@pytest.mark.parametrize("case", ["reuse", "low_confidence", "low_support", "jev_error", "bad_citation"])
def test_live_contract_and_fallback(tmp_path, case):
    r = record()
    requested = []
    def handler(request):
        payload = json.loads(request.content)
        requested.append((request.url.path, payload))
        if request.url.path == "/v1/systemone":
            assert request.headers["authorization"] == "Bearer dummy-jev"
            assert payload["state"]["query"]["conditions"] == {"project": "demo"}
            questions = payload["questions"]
            assert questions["route"]["type"] == "choice"
            assert questions["relevant_" + r.id]["type"] == "noul"
            assert questions["support_" + r.id]["type"] == "score"
            if case == "jev_error":
                return httpx.Response(503)
            return httpx.Response(200, json=jev_reply(r.id,
                confidence=0.1 if case in ("low_confidence", "bad_citation") else 0.98,
                score=1 if case == "low_support" else 2))
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer dummy-llm"
        if len(requested) == 1:
            answer = {"search_text": "报告格式"}
        else:
            answer = {"text": "Markdown", "evidence_ids": ["unknown" if case == "bad_citation" else r.id]}
        return httpx.Response(200, json={"model": "test-generation", "choices": [
            {"message": {"content": json.dumps(answer)}}], "usage": {"output_tokens": 5}})
    settings = Settings(mode="live", llm_key="dummy-llm", jev_key="dummy-jev",
                        retries=0, store_path=tmp_path / "memory.sqlite3")
    agent = MemoryCoordinator(settings, ModelClients(settings, httpx.MockTransport(handler)))
    try:
        agent.memory.add(r)
        result = agent.ask(query())
        if case == "reuse":
            assert result.route == "reuse" and len(requested) == 2
        elif case == "bad_citation":
            assert result.route == "abstain"
        else:
            assert result.route == "synthesize" and len(requested) == 3
        assert len(agent.memory.graph.records) == 1  # generated text never becomes knowledge
    finally:
        agent.close()


def test_retry_and_sanitized_failure():
    attempts = []
    def handler(request):
        attempts.append(request)
        return httpx.Response(429 if len(attempts) == 1 else 401,
                              text="dummy-secret and untrusted provider detail")
    clients = ModelClients(Settings(retries=1), httpx.MockTransport(handler))
    try:
        with pytest.raises(ModelError, match="HTTP 401") as caught:
            clients.generate("answer", "system", {})
        assert len(attempts) == 2
        assert "dummy-secret" not in str(caught.value)
    finally:
        clients.close()


def test_malformed_jev_probabilities_rejected():
    r = record()
    response = jev_reply(r.id)
    response["answers"]["route"]["probabilities"] = {r.id: 0.9}
    clients = ModelClients(Settings(), httpx.MockTransport(
        lambda request: httpx.Response(200, json=response)))
    try:
        with pytest.raises(ModelError, match="invalid"):
            clients.judge(query(), [r])
    finally:
        clients.close()


def test_env_precedence_and_missing_keys(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("MEMORY_AGENT_MODE=live\nMEMORY_LLM_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("MEMORY_LLM_API_KEY", "from-process")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    settings = Settings.load(env)
    assert settings.llm_key == "from-process"
    assert "from-process" not in repr(settings)
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        settings.validate_live()


@pytest.mark.parametrize("failed", [False, True])
def test_probe_checks_both_services_without_writing_memory(failed):
    requested = []
    def handler(request):
        requested.append(request.url.path)
        if failed:
            return httpx.Response(401, text="do not expose provider details")
        if request.url.path.endswith("/systemone"):
            return httpx.Response(200, json=jev_reply("connection_test"))
        return httpx.Response(200, json={"model": "test-generation", "choices": [
            {"message": {"content": '{"ok":true}'}}]})
    settings = Settings(llm_key="dummy-llm", jev_key="dummy-jev", retries=0)
    clients = ModelClients(settings, httpx.MockTransport(handler))
    try:
        report = probe(settings, clients)
        assert report["ok"] is (not failed)
        assert len(requested) == 2
        assert report["memory_written"] is False
        assert "dummy-llm" not in json.dumps(report)
        assert "provider details" not in json.dumps(report)
    finally:
        clients.close()


def test_sqlite_relations_and_integrity(tmp_path):
    store = HypergraphMemory(tmp_path / "memory.sqlite3")
    r = record()
    store.add(r)
    assert store.path.read_bytes()[:16] == b"SQLite format 3\x00"
    with closing(sqlite3.connect(store.path)) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        assert db.execute("SELECT COUNT(*) FROM hyperedge_members WHERE edge_id=?", (r.id,)).fetchone()[0] == 5


def test_short_memory_isolation_override_and_expiry(tmp_path, monkeypatch):
    store = HypergraphMemory(tmp_path / "memory.sqlite3")
    long = record()
    store.add(long)
    expires = now() + timedelta(minutes=5)
    short = long.model_copy(update={"id":"short", "answer":"PDF", "scope":"short",
                                   "session_id":"alice", "valid_until":expires})
    store.add(short)
    qa = query().model_copy(update={"session_id":"alice"})
    qb = query().model_copy(update={"session_id":"bob"})
    assert [r.id for r in store.retrieve(qa)] == [short.id]
    assert [r.id for r in store.retrieve(qb)] == [long.id]
    assert [r.id for r in store.retrieve(query())] == [long.id]
    assert store.get(long.id).superseded_by is None
    monkeypatch.setattr("memory_agent.memory.now", lambda: expires + timedelta(seconds=1))
    assert [r.id for r in store.retrieve(qa)] == [long.id]
    assert store.prune()["expired_short_records_removed"] == 1
    assert len(store.graph.records) == 1
    assert all(n in store.graph.nodes for e in store.graph.edges.values() for n in e.members)


def test_scope_validation_and_unverified_local_cannot_override(tmp_path):
    with pytest.raises(ValidationError):
        Record.model_validate(record().model_dump() | {"scope":"short"})
    store = HypergraphMemory(tmp_path / "memory.sqlite3")
    long = record()
    store.add(long)
    draft = long.model_copy(update={"id":"draft", "scope":"short", "session_id":"alice",
                                   "verified":False, "valid_until":now()+timedelta(minutes=5)})
    store.add(draft)
    assert [r.id for r in store.retrieve(query().model_copy(update={"session_id":"alice"}))] == [long.id]


def test_batch_transaction_rolls_back_supersession(tmp_path):
    store = HypergraphMemory(tmp_path / "memory.sqlite3")
    old = record()
    store.add(old)
    new = old.model_copy(update={"id":"new", "answer":"PDF", "created_at":now()})
    before = store.counts()
    with pytest.raises(ValueError, match="already exists"):
        store.add_many([new, old])
    assert store.counts() == before
    assert store.get(old.id).superseded_by is None


def test_multiple_store_instances_do_not_overwrite_updates(tmp_path):
    path = tmp_path / "memory.sqlite3"
    first, second = HypergraphMemory(path), HypergraphMemory(path)
    def add(i):
        r = record().model_copy(update={"id":f"r{i}", "key":f"key{i}"})
        (first if i % 2 else second).add(r)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(add, range(8)))
    assert first.counts()["records"] == second.counts()["records"] == 8


def test_explicit_promotion_keeps_source_and_updates_long_memory(tmp_path):
    store = HypergraphMemory(tmp_path / "memory.sqlite3")
    long = record()
    store.add(long)
    short = long.model_copy(update={"id":"short", "answer":"PDF", "scope":"short",
                                   "session_id":"alice", "valid_until":now()+timedelta(minutes=5)})
    store.add(short)
    promoted_id = store.promote(short.id, "user explicitly confirmed permanent preference")
    assert store.get(promoted_id).scope == "long"
    assert store.get(promoted_id).session_id is None
    assert store.get(promoted_id).valid_until is None
    assert store.get(short.id).superseded_by == promoted_id
    assert store.get(long.id).superseded_by == promoted_id
    assert store.retrieve(query())[0].answer == "PDF"
    with pytest.raises(ValueError):
        store.promote(short.id, "repeat")


def test_session_history_persistence_limit_and_expiry(tmp_path, monkeypatch):
    settings = Settings(store_path=tmp_path / "memory.sqlite3", session_turn_limit=2,
                        short_ttl_minutes=1)
    agent = MemoryCoordinator(settings)
    try:
        agent.memory.add(record())
        for i in range(3):
            q = query().model_copy(update={"session_id":"alice", "question":f"question {i}"})
            agent.ask(q)
        loaded = HypergraphMemory(settings.store_path)
        assert [t["question"] for t in loaded.history("alice")] == ["question 1", "question 2"]
        assert loaded.history("bob") == []
        assert loaded.counts()["records"] == 1  # dialogue never becomes verified knowledge
        future = now()+timedelta(minutes=2)
        monkeypatch.setattr("memory_agent.memory.now", lambda: future)
        assert loaded.history("alice") == []
        assert loaded.prune()["expired_turns_removed"] == 2
    finally:
        agent.close()


def test_live_understanding_receives_only_current_session_history(tmp_path):
    seen = []
    def handler(request):
        payload = json.loads(request.content)
        seen.append(json.loads(payload["messages"][1]["content"]))
        return httpx.Response(200, json={"choices":[{"message":{"content":'{"search_text":"format"}'}}]})
    settings = Settings(mode="live", llm_key="dummy", jev_key="dummy", retries=0,
                        store_path=tmp_path / "memory.sqlite3")
    agent = MemoryCoordinator(settings, ModelClients(settings, httpx.MockTransport(handler)))
    try:
        # No evidence: only understanding call, then abstain and store dialogue.
        agent.ask(Query(question="previous alice",session_id="alice"))
        agent.ask(Query(question="previous bob",session_id="bob"))
        agent.ask(Query(question="this question",session_id="alice"))
        assert [t["question"] for t in seen[-1]["recent_turns"]] == ["previous alice"]
        assert agent.memory.counts()["records"] == 0
    finally:
        agent.close()


def test_json_migration_preserves_graph_status_and_is_idempotent(tmp_path):
    source = HypergraphMemory(tmp_path / "memory.sqlite3")
    r = record()
    source.add(r)
    source.fail(r.id, "old feedback")
    payload = source.graph.model_dump(mode="json")
    for item in payload["records"].values():
        item.pop("scope")
        item.pop("session_id")
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(payload),encoding="utf-8")
    # Reset the SQLite store, only after capturing legacy data.
    with source.connection(write=True) as db:
        db.execute("DELETE FROM hyperedges")
        db.execute("DELETE FROM nodes")
        db.execute("DELETE FROM records")
    assert source.migrate_json(legacy)["imported"] == 1
    assert source.get(r.id).failures == ["old feedback"]
    assert not source.retrieve(query())
    assert source.migrate_json(legacy)["already_migrated"] is True
    assert legacy.exists()


def test_invalid_migration_leaves_database_empty(tmp_path):
    store = HypergraphMemory(tmp_path / "memory.sqlite3")
    payload = {"version":1,"records":{},"nodes":{},"edges":{
        "broken":{"id":"broken","kind":"episode","members":["missing"],"record_ids":[]}}}
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises(ValueError,match="dangling"):
        store.migrate_json(legacy)
    assert store.counts()["records"] == 0
    assert store.counts()["hyperedges"] == 0
