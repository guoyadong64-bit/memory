import httpx
import pytest

from benchmarks.run_budget import BudgetedClients, BudgetExceeded, RunBudget
from memory_agent.clients import ModelError
from memory_agent.settings import Settings


def test_budget_persists_between_clients_and_stops_before_next_http(tmp_path):
    sent = []
    def handler(request):
        sent.append(request)
        return httpx.Response(200, json={"usage": {"input_tokens": 7, "output_tokens": 3}})
    budget = RunBudget(tmp_path, max_requests=2, max_reported_tokens=10)
    client = BudgetedClients(Settings(), budget, httpx.MockTransport(handler))
    client.post("https://example.test", "dummy", {"model": "test"}, "jev")
    resumed = RunBudget(tmp_path, max_requests=2, max_reported_tokens=10)
    second = BudgetedClients(Settings(), resumed, httpx.MockTransport(handler))
    try:
        with pytest.raises(BudgetExceeded, match="token"):
            second.post("https://example.test", "dummy", {"model": "test"}, "answer")
        assert len(sent) == 1
        assert resumed.state["reported_tokens"] == 10
    finally:
        client.close()
        second.close()


def test_failed_requests_count_and_cannot_be_hidden_by_agent_fallback(tmp_path):
    budget = RunBudget(tmp_path, max_requests=1)
    client = BudgetedClients(Settings(retries=2), budget, httpx.MockTransport(
        lambda request: httpx.Response(503, text="untrusted")))
    try:
        with pytest.raises(ModelError):
            client.post("https://example.test", "dummy", {"model": "test"}, "answer")
        with pytest.raises(BudgetExceeded):
            client.post("https://example.test", "dummy", {"model": "test"}, "answer")
        assert budget.state["failed_requests"] == 1
        assert budget.state["requests_started"] == 1
        assert not issubclass(BudgetExceeded, ModelError)
    finally:
        client.close()


def test_explicit_extension_keeps_prior_usage_and_is_resumable(tmp_path):
    budget = RunBudget(tmp_path, max_reported_tokens=10)
    budget.begin()
    budget.finish("test", "model", {"total_tokens": 10})
    budget.extend_token_limit(20, "Explicit user approval in test fixture")
    resumed = RunBudget(tmp_path)
    assert resumed.state["reported_tokens"] == 10
    assert resumed.state["requests_started"] == 1
    assert resumed.limits["max_reported_tokens"] == 20
    assert len(resumed.state["limit_changes"]) == 1


def test_authorized_unlimited_usage_keeps_counters_and_request_bound(tmp_path):
    budget = RunBudget(tmp_path, max_requests=2, max_reported_tokens=10)
    budget.begin()
    budget.finish("test", "model", {"total_tokens": 10})
    budget.remove_token_limit("Explicit user authorization to remove cumulative cap")
    resumed = RunBudget(tmp_path)
    assert resumed.limits["max_reported_tokens"] == 0
    assert resumed.state["reported_tokens"] == 10
    resumed.begin()
    resumed.finish("test", "model", {"total_tokens": 100})
    assert resumed.state["reported_tokens"] == 110
    with pytest.raises(BudgetExceeded, match="request"):
        resumed.begin()


def test_independent_workers_share_accounting_without_lost_updates(tmp_path):
    import json
    from concurrent.futures import ThreadPoolExecutor
    budget=RunBudget(tmp_path,max_requests=40,max_reported_tokens=0)
    def call(index):
        budget.begin();budget.finish('test','model',{'total_tokens':index+1})
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(call,range(40)))
    assert budget.state['requests_started']==40
    assert budget.state['reported_tokens']==sum(range(1,41))
    logs=[json.loads(x) for x in budget.log.read_text().splitlines()]
    assert len(logs)==40 and sum(x['reported_tokens'] for x in logs)==sum(range(1,41))
