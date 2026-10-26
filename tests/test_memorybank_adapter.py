"""Execute pinned MemoryBank code with real FAISS, but never real model APIs."""
import inspect
import json
from pathlib import Path
import shutil
from uuid import uuid4

import httpx
import pytest

np = pytest.importorskip("numpy")
faiss = pytest.importorskip("faiss")

from benchmarks import memorybank_adapter as adapter
from benchmarks import run_validation50_memorybank as runner
from benchmarks.comparison_worker import CachedClients, atomic_json
from benchmarks.run_budget import RunBudget
from memory_agent.settings import Settings


@pytest.fixture
def workspace():
    root = Path(__file__).resolve().parents[1] / "runtime/memorybank_tests" / uuid4().hex
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root)


def observations(count=3):
    return [{"frame_index": n, "clock": f"06:00:{n:02d}", "description": f"Cup observation {n}",
        "entities": [{"id": "cup", "name": "cup", "kind": "object", "labels": [], "attributes": {}}],
        "relations": [], "interaction": None, "uncertainty": []} for n in range(count)]


def client_for(directory, sent, budget=None):
    def respond(request):
        payload = json.loads(request.content)
        sent.append(payload)
        assert "gold" not in payload and "gold_goal_frames" not in payload
        if "input" in payload:
            data = [{"index": i, "embedding": [1.0, 1.0 + len(text) % 7, .5]}
                    for i, text in enumerate(payload["input"])]
            return httpx.Response(200, json={"data": data, "usage": {"total_tokens": 2}})
        if payload["max_tokens"] == 400:
            content = "A cup was observed. Frame 95 is mentioned but is not a source-frame citation."
        else:
            state = json.loads(payload["messages"][1]["content"][0]["text"])
            evidence_id = next(item["id"] for item in state["evidence"] if 0 in item["frame_indices"])
            content = json.dumps({"goal_frames": [0], "evidence_ids": [evidence_id]})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                       "usage": {"total_tokens": 10}})
    budget = budget or RunBudget(directory, max_requests=1000, max_reported_tokens=0)
    client = CachedClients(Settings(llm_key="fake", llm_model="fake", embedding_model="fake"),
        budget, directory / "response_cache", transport=httpx.MockTransport(respond))
    return client, budget


def test_pinned_definitions_execute_from_official_files():
    root, audit = adapter.official_source()
    namespace = adapter.upstream_namespace(root)
    assert audit["revision"] == adapter.REVISION
    assert namespace["similarity_search_with_score_by_vector"].__code__.co_filename == str(
        root / "memory_bank/memory_retrieval/local_doc_qa.py")
    assert namespace["summarize_content_prompt"].__code__.co_filename == str(root / "memory_bank/summarize_memory.py")


def test_official_file_change_is_rejected_before_requests(monkeypatch):
    monkeypatch.setattr(adapter, "SOURCE_HASHES", {"memory_bank/summarize_memory.py": "0" * 64})
    with pytest.raises(RuntimeError, match="官方文件被修改"):
        adapter.OfficialMemoryBank(None)


def test_query_order_cannot_mutate_original_documents_and_tracks_exact_merge_sources():
    agent = adapter.OfficialMemoryBank(None)
    docs = [adapter.Document(adapter.WitnessText(f"short frame {n}", [str(n)]), {"source": "day"}) for n in range(5)]
    index = faiss.IndexFlatL2(2)
    index.add(np.asarray([[1, n + 1] for n in range(5)], dtype=np.float32))
    before = [str(doc.page_content) for doc in docs]
    first = agent.search(index, docs, [1, 1])
    agent.search(index, docs, [1, 5])
    again = agent.search(index, docs, [1, 1])
    assert [str(doc.page_content) for doc in docs] == before
    assert [(str(doc.page_content), score) for doc, score in first] == [(str(doc.page_content), score) for doc, score in again]
    assert set().union(*(set(doc.page_content.source_ids) for doc, _ in first)) == {str(n) for n in range(5)}


def test_build_resume_and_summary_do_not_expose_unretrieved_frames(workspace):
    sent = []
    client, budget = client_for(workspace, sent)
    frames = observations()
    memory = adapter.MemoryBankMethod(workspace, workspace / "shared", {"ep": frames}, client, lambda **event: None)
    memory.build()
    assert len(sent) == 2  # One official day-summary request and one embedding batch.
    index, docs, provenance = memory.banks["ep"]
    assert index.ntotal == 4 and provenance["ep_summary"] == []
    assert "95" in str(docs[-1].page_content)
    evidence, resolution, details = memory.retrieve("ep", "Navigate to a cup.")
    assert resolution is None
    assert {n for item in evidence for n in item["frame_indices"]} == {0, 1, 2}
    assert details["forgetting_enabled"] is False
    assert len(evidence) <= 20 and len(json.dumps(evidence, ensure_ascii=False).encode()) <= 40000
    requests = budget.state["requests_started"]
    resumed = adapter.MemoryBankMethod(workspace, workspace / "shared", {"ep": frames}, client, lambda **event: None)
    resumed.build()
    resumed.retrieve("ep", "Navigate to a cup.")
    assert budget.state["requests_started"] == requests == len(sent)
    changed = [dict(frame) for frame in frames]
    changed[0]["description"] = "Changed original observation"
    with pytest.raises(RuntimeError, match="different observations"):
        adapter.MemoryBankMethod(workspace, workspace / "shared", {"ep": changed}, client, lambda **event: None).build()
    assert budget.state["requests_started"] == requests
    client.close()


def test_namespace_isolation_and_no_question_writeback(workspace):
    sent = []
    client, _ = client_for(workspace, sent)
    memory = adapter.MemoryBankMethod(workspace, workspace / "shared", {"ep_a": observations(), "ep_b": observations()},
                                     client, lambda **event: None)
    memory.build()
    files = {ep: (workspace / "memorybank" / ep / "memory_bank.json").read_bytes() for ep in memory.banks}
    for ep in memory.banks:
        evidence, _, diagnostics = memory.retrieve(ep, "Navigate to a cup.")
        assert all(identity.startswith(ep + "_") for item in diagnostics["sources"].values()
                   for identity in item["document_ids"])
        assert all(item["id"].startswith(ep + "_") for item in evidence)
    assert all((workspace / "memorybank" / ep / "memory_bank.json").read_bytes() == before
               for ep, before in files.items())
    assert list(inspect.signature(memory.retrieve).parameters) == ["episode", "question"]
    client.close()


def test_full_runner_uses_common_visual_decoder_and_resumes_without_new_calls(workspace):
    shared, run = workspace / "shared", workspace / "experiment"
    atomic_json(shared / "perception/api_budget.json", {"requests_started": 0, "reported_tokens": 0})
    for n in range(3):
        path = shared / "frames/ep_1" / f"{n:03d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"offline mock image")
    tasks = [{"row_idx": n, "ep_id": "ep_1", "task_id": f"task_{n}", "question": "Navigate to a cup.",
              "answer": "[[0]]", "high_level_category": "Single-Goal Spatial Tasks",
              "low_level_category": "Object Recall"} for n in range(2)]
    sent = []
    def factory(settings, budget, cache):
        # Use the same cached clients and mock HTTP responder as adapter tests.
        client, _ = client_for(run, sent, budget=budget)
        return client
    result = runner.evaluate(run, shared, tasks, {"ep_1": observations()},
        Settings(llm_key="fake", llm_model="fake", embedding_model="fake"), client_factory=factory)
    assert result["complete"] and result["correct"] == result["total"] == 2
    calls = len(sent)
    repeated = runner.evaluate(run, shared, tasks, {"ep_1": observations()},
        Settings(llm_key="fake", llm_model="fake", embedding_model="fake"), client_factory=factory)
    assert repeated["api_usage"]["requests_started"] == calls == len(sent)
    assert (run / "REPORT.md").read_text(encoding="utf-8").startswith("# MemoryBank")
    rows = [json.loads(line) for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and all(row["method"] == "memorybank" for row in rows)


def test_interrupted_memory_build_reuses_cached_paid_embedding_response(workspace, monkeypatch):
    sent = []
    client, budget = client_for(workspace, sent)
    original_post = client.post
    interrupt = [True]
    def post(url, key, payload, role):
        response = original_post(url, key, payload, role)
        if role == "embedding" and interrupt[0]:
            interrupt[0] = False
            raise RuntimeError("Interrupted after paid embedding response")
        return response
    monkeypatch.setattr(client, "post", post)
    def memory():
        return adapter.MemoryBankMethod(workspace, workspace / "shared", {"ep": observations()},
                                        client, lambda **event: None)
    with pytest.raises(RuntimeError, match="Interrupted after paid"):
        memory().build()
    assert budget.state["requests_started"] == len(sent) == 2
    assert (workspace / "memorybank/ep/summary_checkpoint.json").exists()
    assert not (workspace / "memorybank/ep/embedding_checkpoint.json").exists()
    resumed = memory()
    resumed.build()
    assert resumed.banks["ep"][0].ntotal == 4
    assert budget.state["requests_started"] == len(sent) == 2
    client.close()
