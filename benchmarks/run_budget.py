"""Persistent limits shared by perception and both retrieval strategies.

Reported-token limits stop subsequent requests; one response can cross the threshold.
Failed requests still consume the request allowance even if usage is not returned.
"""
from datetime import datetime, timezone
import json
from threading import RLock
import time

from memory_agent.clients import ModelClients


class BudgetExceeded(RuntimeError):
    """Must propagate through agent fallback handlers and stop evaluation."""


class RunBudget:
    def __init__(self, run, max_requests=None, max_reported_tokens=None):
        self._lock = RLock()
        self.path = run / "api_budget.json"
        self.log = run / "api_requests.jsonl"
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if max_requests is None:
                max_requests = self.state["limits"]["max_requests"]
            if max_reported_tokens is None:
                max_reported_tokens = self.state["limits"]["max_reported_tokens"]
            self.limits = {"max_requests": max_requests, "max_reported_tokens": max_reported_tokens}
            if self.state["limits"] != self.limits:
                raise BudgetExceeded("Budget configuration changed; use a separate run directory")
        else:
            self.limits = {"max_requests": max_requests if max_requests is not None else 260,
                           "max_reported_tokens": max_reported_tokens if max_reported_tokens is not None else 1_200_000}
            self.state = {"limits": self.limits, "requests_started": 0,
                          "reported_tokens": 0, "failed_requests": 0}
            self.save()

    def extend_token_limit(self, new_limit, reason):
        """Call only after explicit user authorization; retain all prior counters."""
        if type(new_limit) is not int or self.limits["max_reported_tokens"] == 0 or new_limit <= self.limits["max_reported_tokens"]:
            raise ValueError("New token limit must exceed the existing limit")
        self.state.setdefault("limit_changes", []).append({
            "at_utc": datetime.now(timezone.utc).isoformat(), "old": self.limits["max_reported_tokens"],
            "new": new_limit, "reason": reason})
        self.limits["max_reported_tokens"] = new_limit
        self.state["limits"] = self.limits
        self.save()

    def remove_token_limit(self, reason):
        """Explicitly authorized unlimited cumulative usage; keep accounting intact.

        Zero is the persisted unlimited sentinel, so new dependent stages can
        inherit it without conflating it with the constructor's default None.
        """
        if not reason.strip():
            raise ValueError("Authorization reason required")
        if self.limits["max_reported_tokens"] == 0:
            return
        self.state.setdefault("limit_changes", []).append({
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "old": self.limits["max_reported_tokens"], "new": 0, "reason": reason})
        self.limits["max_reported_tokens"] = 0
        self.state["limits"] = self.limits
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")
        # On Windows, another process briefly reading api_budget.json can make
        # os.replace raise WinError 5/32. Preserve the same temp file and retry
        # only the atomic rename; no model request is retried by this path.
        for attempt in range(50):
            try:
                temp.replace(self.path)
                return
            except PermissionError:
                if attempt == 49:
                    raise
                time.sleep(0.1)

    def begin(self):
        with self._lock:
            self._begin_locked()

    def _begin_locked(self):
        if self.state["requests_started"] >= self.limits["max_requests"]:
            raise BudgetExceeded("Cumulative API request limit reached; experiment paused without another call")
        if self.limits["max_reported_tokens"] and self.state["reported_tokens"] >= self.limits["max_reported_tokens"]:
            raise BudgetExceeded("Cumulative reported-token limit reached; experiment paused without another call")
        self.state["requests_started"] += 1
        self.save()  # count even if interrupted during HTTP

    def finish(self, role, model, usage, error=None):
        with self._lock:
            self._finish_locked(role,model,usage,error)

    def _finish_locked(self, role, model, usage, error=None):
        tokens = usage.get("total_tokens", usage.get("prompt_tokens", usage.get("input_tokens", 0)) +
                           usage.get("completion_tokens", usage.get("output_tokens", 0)))
        self.state["reported_tokens"] += tokens
        self.state["failed_requests"] += int(error is not None)
        self.save()
        entry = {"finished_at_utc": datetime.now(timezone.utc).isoformat(), "role": role,
                 "model": model, "usage": usage, "reported_tokens": tokens, "error": error}
        with self.log.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")


class BudgetedClients(ModelClients):
    def __init__(self, settings, budget, transport=None):
        super().__init__(settings.model_copy(update={"retries": 0}), transport)
        self.budget = budget

    def post(self, url, key, payload, role):
        # Retry at most one ambiguous remote disconnect, plus at most two
        # connection-establishment failures. Each HTTP attempt receives its own
        # ledger entry. A server may have processed the disconnected request.
        remote_replays = 0
        connect_retries = 0
        for _ in range(4):
            self.budget.begin()
            try:
                result = super().post(url, key, payload, role)
            except Exception as exc:
                kind = getattr(exc, "transport_kind", "")
                self.budget.finish(role, payload["model"], {},
                                   type(exc).__name__ + (":" + kind if kind else ""))
                if self.settings.retries == 0:
                    if kind == "RemoteProtocolError" and remote_replays < 1:
                        remote_replays += 1
                        time.sleep(2)
                        continue
                    if kind in {"ConnectError", "ConnectTimeout"} and connect_retries < 2:
                        connect_retries += 1
                        time.sleep(2 ** connect_retries)
                        continue
                raise
            self.budget.finish(role, result.get("model", payload["model"]), result.get("usage", {}))
            return result
