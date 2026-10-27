"""Offline tests: no live model traffic, including the interrupted/resumed pipeline."""
from collections import Counter
import ast
import json
import re
import string
from pathlib import Path

import httpx
import pytest

from benchmarks import run_locomo_hypergraph as runner
from benchmarks.locomo_data import (DEFAULT_DATA, OfficialScorer, conversation_view,
                                    digest, select_tasks)
from benchmarks.locomo_hypergraph import Events, LoCoMoHypergraph, event_record
from memory_agent.settings import Settings


def sample():
    return {"sample_id": "conv-test", "conversation": {
        "speaker_a": "Ana", "speaker_b": "Bo",
        "session_1_date_time": "10:00 am on 8 May, 2023",
        "session_1": [{"dia_id": "D1:1", "speaker": "Ana", "text": "I moved to Paris yesterday."},
                      {"dia_id": "D1:2", "speaker": "Bo", "text": "Paris sounds wonderful."}],
        "session_2_date_time": "10:00 am on 9 May, 2023",
        "session_2": [{"dia_id": "D2:1", "speaker": "Ana", "text": "I bought a bicycle."}]},
        "event_summary": "GOLD_SUMMARY_NOT_ALLOWED", "observation": "PRECOMPUTED_NOT_ALLOWED",
        "qa": [
            {"question": "Where did Ana move?", "answer": "Paris", "evidence": ["D1:1"], "category": 4},
            {"question": "When did Ana move?", "answer": "7 May 2023", "evidence": ["D1:1"], "category": 2},
            {"question": "List Ana's city and purchase.", "answer": "Paris, bicycle", "evidence": ["D1:1", "D2:1"], "category": 1},
            {"question": "What is the purchase?", "answer": "bicycle; a bike", "evidence": ["D2:1"], "category": 3},
            {"question": "What is Ana's secret?", "adversarial_answer": "GOLD_SECRET_NOT_ALLOWED", "evidence": [], "category": 5}]}


@pytest.fixture
def scorer():
    if not (DEFAULT_DATA / "official/evaluation.py").exists():
        pytest.skip("Prepare pinned public LoCoMo scorer with --prepare-only")
    return OfficialScorer(DEFAULT_DATA)


def settings():
    return Settings(llm_key="test-key", llm_model="mock-model", embedding_model="mock-embedding",
                    llm_url="https://offline.test/v1", mode="live", retries=0)


def config():
    options = {"retrieval": {"batch_size": 20, "top_k": 6, "evidence_records": 12, "evidence_chars": 16000}}
    return {"options": options, "fingerprint": digest(options)}


def transport(sent, *, invalid_source=False):
    def handle(request):
        payload = json.loads(request.content)
        sent.append(payload)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"model": payload["model"], "usage": {"total_tokens": 3},
                "data": [{"index": index, "embedding": [1, len(text) % 17 + 1, len(text) % 7 + 1]}
                         for index, text in enumerate(payload["input"])]})
        state = json.loads(payload["messages"][1]["content"])["input"]
        if "current_turns" in state:
            first = state["current_turns"][0]
            result = {"events": [{"summary": first["text"], "source_ids": ["D99:999" if invalid_source else first["dia_id"]],
                                   "entities": [{"name": "Ana", "kind": "agent"}, {"name": "Paris", "kind": "place"}],
                                   "event_time": "yesterday"}]}
        else:
            answers = {"Where did Ana move?": "Paris", "When did Ana move?": "7 May 2023",
                       "List Ana's city and purchase.": "Paris, bicycle", "What is the purchase?": "bicycle",
                       "What is Ana's secret?": "not mentioned"}
            result = {"answer": answers[state["question"]], "evidence_ids": []}
        return httpx.Response(200, json={"model": payload["model"], "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}]})
    return httpx.MockTransport(handle)


def clients(sent, **kwargs):
    def create(settings, budget, cache):
        return runner.LoCoMoClients(settings, budget, cache, transport=transport(sent, **kwargs))
    return create


def test_conversation_input_excludes_all_annotations_and_keeps_dates():
    view = conversation_view(sample())
    text = json.dumps(view)
    assert "GOLD_" not in text and "PRECOMPUTED_" not in text and "qa" not in view
    assert view["turns"][0]["observed_at"] == "2023-05-08T10:00:00+00:00"
    assert view["turns"][1]["observed_at"] == "2023-05-08T10:00:00.000001+00:00"
    views, tasks = select_tasks([sample()])
    assert len(views) == 1 and len(tasks) == 4
    assert [task["category"] for task in tasks] == [4, 2, 1, 3]
    assert len(select_tasks([sample()], categories=(5,))[1]) == 1
    assert len(select_tasks([sample()], max_questions=1)[1]) == 1


def test_official_scorer_matches_full_upstream_function(scorer):
    # Run the actual upstream wrapper without importing unrelated BERTScore.
    from nltk.stem import PorterStemmer
    tree = ast.parse((DEFAULT_DATA / "official/evaluation.py").read_text(encoding="utf-8"))
    keep = {"normalize_answer", "f1_score", "f1", "eval_question_answering"}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in keep]
    class Mean:
        @staticmethod
        def mean(values): return sum(values) / len(values)
    namespace = {"Counter": Counter, "regex": re, "string": string, "ps": PorterStemmer(), "np": Mean}
    exec(compile(ast.Module(body=body, type_ignores=[]), "upstream", "exec"), namespace)
    cases = sample()["qa"][:4] + [{"question": "Unknown", "answer": "", "category": 5, "evidence": []}]
    predictions = ["The Paris", "7 May", "Paris, a bicycle", "bicycles", "not mentioned"]
    qa = [dict(case, prediction=prediction) for case, prediction in zip(cases, predictions)]
    expected, _, _ = namespace["eval_question_answering"](qa)
    assert [scorer.score(prediction, case) for prediction, case in zip(predictions, cases)] == pytest.approx(expected)
    assert scorer.evidence_recall(cases[2], ["D1:1"]) == .5
    assert scorer.evidence_recall(cases[4], []) is None


def test_full_pipeline_persisted_graph_no_gold_and_resume_no_billing(tmp_path, scorer):
    views, tasks = select_tasks([sample()])
    sent = []
    result = runner.evaluate(tmp_path, views, tasks, settings(), config(), scorer, client_factory=clients(sent))
    assert result["complete"] and result["completed"] == 4 and result["overall_f1"] == 1
    assert result["gold_dialog_evidence_recall"] == 1
    assert len(list((tmp_path / "answers").glob("*.json"))) == 4
    assert len((tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 4
    assert (tmp_path / "memory/conv-test.sqlite3").exists()
    import sqlite3
    with sqlite3.connect(tmp_path / "memory/conv-test.sqlite3") as db:
        kinds = {row[0] for row in db.execute("SELECT DISTINCT kind FROM hyperedges")}
        assert {"episode", "family", "entity_identity", "observation", "process", "stage_evidence", "role_binding"} <= kinds
        shared = db.execute("SELECT n.id,count(DISTINCT er.record_id) FROM nodes n "
                            "JOIN hyperedge_members m ON m.node_id=n.id JOIN hyperedges e ON e.id=m.edge_id "
                            "JOIN hyperedge_records er ON er.edge_id=e.id "
                            "WHERE n.kind='entity' AND e.kind='episode' GROUP BY n.id HAVING count(DISTINCT er.record_id)>1").fetchall()
        assert shared
    initial_calls = len(sent)
    second = runner.evaluate(tmp_path, views, tasks, settings(), config(), scorer, client_factory=clients(sent))
    assert second["api_usage"]["requests_started"] == result["api_usage"]["requests_started"] == initial_calls
    assert len(sent) == initial_calls
    assert "GOLD_" not in json.dumps(sent) and "PRECOMPUTED_" not in json.dumps(sent)
    for payload in sent:
        if "messages" in payload:
            state = json.loads(payload["messages"][1]["content"])["input"]
            assert not {"qa", "answer", "category", "reference_answer", "event_summary"} & state.keys()
    assert "100.00" in (tmp_path / "REPORT.md").read_text(encoding="utf-8")


def test_crash_after_paid_answer_reuses_cache_on_resume(tmp_path, scorer, monkeypatch):
    views, tasks = select_tasks([sample()], max_questions=1)
    sent, fail = [], [True]
    original_write = runner.atomic_json
    def interrupted(path, value):
        if Path(path).parent.name == "answers" and fail[0]:
            fail[0] = False
            raise RuntimeError("simulated interruption after model response")
        return original_write(path, value)
    monkeypatch.setattr(runner, "atomic_json", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.evaluate(tmp_path, views, tasks, settings(), config(), scorer, client_factory=clients(sent))
    before = len(sent)
    assert runner.load(tmp_path / "summary.json")["complete"] is False
    result = runner.evaluate(tmp_path, views, tasks, settings(), config(), scorer, client_factory=clients(sent))
    assert result["complete"] and len(sent) == before
    assert result["api_usage"]["requests_started"] == before


def test_invalid_source_stops_without_silent_fact_invention(tmp_path, scorer):
    views, tasks = select_tasks([sample()], max_questions=1)
    sent = []
    with pytest.raises(Exception, match="structured output invalid"):
        runner.evaluate(tmp_path, views, tasks, settings(), config(), scorer,
                        client_factory=clients(sent, invalid_source=True))
    assert len(sent) == 3
    assert len(list((tmp_path / "format_failures").rglob("round_*_attempt_*.json"))) == 3
    assert not list((tmp_path / "answers").glob("*.json"))
    # Raw dialogue survives an invalid extracted summary.
    import sqlite3
    with sqlite3.connect(tmp_path / "memory/conv-test.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM records").fetchone()[0] == 3
    result = runner.evaluate(tmp_path, views, tasks, settings(), config(), scorer, client_factory=clients(sent))
    assert result["complete"] and result["overall_f1"] == 1
    assert len(sent) > 3  # fresh recovery round, not an endless replay of invalid cached JSON


def test_changed_config_and_foreign_answers_are_rejected_before_models(tmp_path, scorer):
    frozen = runner.freeze_config(tmp_path, {"model": "first"})
    with pytest.raises(RuntimeError, match="模型"):
        runner.freeze_config(tmp_path, {"model": "second"})
    views, tasks = select_tasks([sample()], max_questions=1)
    runner.atomic_json(tmp_path / "answers/foreign.json", {"task_id": "foreign"})
    with pytest.raises(RuntimeError, match="答题断点"):
        runner.load_answers(tmp_path, tasks, frozen, scorer)


def test_no_fabricated_event_datetime_from_relative_expression():
    view = conversation_view(sample())
    event = Events.model_validate({"events": [{"summary": "Ana planned a trip next year.",
        "source_ids": ["D1:1"], "entities": [], "event_time": "next year"}]}).events[0]
    record = event_record(view, event, 0, 0)
    assert record.time.event_at.isoformat() == "2023-05-08T10:00:00+00:00"
    assert json.loads(record.answer)["event_time"] == "next year"
    assert record.evolution.stages[0].phase == "reported"


def test_adversarial_scoring_not_mixed_into_main_f1(tmp_path, scorer):
    views, tasks = select_tasks([sample()], categories=(4, 5))
    result = runner.evaluate(tmp_path, views, tasks, settings(), config(), scorer, client_factory=clients([]))
    assert result["overall_f1"] == 1 and result["overall_f1_questions"] == 1
    assert result["by_category"]["5"]["metric"] == "abstention_accuracy"
    assert result["by_category"]["5"]["score"] == 1
