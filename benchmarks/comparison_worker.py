"""Run one registered memory method in its own Python environment."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmarks.comparison_methods import REGISTRY
from benchmarks.findingdory_pilot import load, score_frames
from benchmarks.findingdory_visual_answer import decode_visual
from benchmarks.run_budget import BudgetedClients, RunBudget
from memory_agent.settings import Settings


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


class CachedClients(BudgetedClients):
    """Durable response cache prevents billing completed requests after resumption."""

    def __init__(self, settings, budget, cache_dir, transport=None):
        super().__init__(settings, budget, transport)
        self.cache_dir = Path(cache_dir)

    def post(self, url, key, payload, role):
        digest = hashlib.sha256(json.dumps(
            {"url": url, "role": role, "payload": payload}, sort_keys=True,
            ensure_ascii=False).encode()).hexdigest()
        path = self.cache_dir / f"{digest}.json"
        if path.exists():
            self.calls.append({"role": role, "cached": True, "model": payload.get("model")})
            return load(path)
        result = super().post(url, key, payload, role)
        atomic_json(path, result)
        return result


def emit(budget, **event):
    state = budget.state
    print("@@PROGRESS@@" + json.dumps({**event, "requests": state["requests_started"],
          "tokens": state["reported_tokens"]}, ensure_ascii=False), flush=True)


def score_task(task, answer, evidence):
    gold = json.loads(task["answer"])
    predicted = answer.get("goal_frames", [])
    correct = score_frames(predicted, gold,
                           task["low_level_category"] == "Unordered Revisitation")
    candidate = {index for item in evidence for index in item["frame_indices"]}
    coverage = sum(bool(candidate.intersection(group)) for group in gold) / len(gold)
    return correct, coverage


def run_method(run: Path, data: Path, name: str, plugins=()):
    for plugin in plugins:
        importlib.import_module(plugin)
    if name not in REGISTRY:
        raise ValueError(f"Unknown comparison method: {name}")
    root = run / "methods" / name
    budget = RunBudget(root, max_requests=5000, max_reported_tokens=0)
    settings = Settings.load().model_copy(update={"mode": "live", "timeout": 60, "retries": 0})
    client = CachedClients(settings, budget, root / "response_cache")
    manifest, tasks = load(data / "manifest.json"), load(data / "selected_tasks.json")
    observations = {ep: load(run / "observations" / f"{ep}.json")["observations"]
                    for ep in manifest["episodes"]}
    method = REGISTRY[name](root, data, observations, client,
                            lambda **event: emit(budget, method=name, **event))
    answers = root / "answers"
    try:
        method.build()
        for n, task in enumerate(tasks, 1):
            checkpoint = answers / f"{task['row_idx']}.json"
            if checkpoint.exists():
                row = load(checkpoint)
                if row["task_id"] != task["task_id"]:
                    raise RuntimeError("Task checkpoint does not match frozen data")
            else:
                started = time.monotonic()
                client.calls.clear()
                evidence, resolution, retrieval = method.retrieve(task["ep_id"], task["question"])
                answer = decode_visual(client, task["question"], evidence,
                                       observations[task["ep_id"]], data, task["ep_id"], resolution)
                correct, covered = score_task(task, answer, evidence)
                row = {"method": name, "row_idx": task["row_idx"], "task_id": task["task_id"],
                       "episode": task["ep_id"], "question": task["question"],
                       "prediction": answer, "correct": correct,
                       "candidate_frame_covered": covered,
                       "evidence": evidence, "resolution": resolution,
                       "retrieval": retrieval, "calls": client.calls.copy(),
                       "latency_seconds": round(time.monotonic()-started, 3)}
                atomic_json(checkpoint, row)
            completed = [load(p) for p in answers.glob("*.json")]
            emit(budget, stage="answer", method=name, episode=task["ep_id"],
                 completed=len(completed), total=len(tasks),
                 correct=sum(item["correct"] for item in completed))
    finally:
        try:
            method.close()
        finally:
            client.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--plugin", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        run_method(args.run_dir, args.data_dir, args.method, args.plugin)
    except BaseException as exc:
        message = str(exc)
        try:
            settings = Settings.load()
            for secret in (settings.llm_key, settings.embedding_key, settings.jev_key):
                if secret and len(secret) >= 8:
                    message = message.replace(secret, "[REDACTED]")
        except Exception:
            message = type(exc).__name__
        atomic_json(args.run_dir / "methods" / args.method / "last_failure.json",
                    {"type": type(exc).__name__, "message": message[:500]})
        print(f"方法 {args.method} 失败：{type(exc).__name__}: {message[:500]}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
