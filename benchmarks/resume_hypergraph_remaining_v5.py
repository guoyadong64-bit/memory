"""Resume the existing V5 run, adding bounded recovery for invalid JSON only.

Original source, config, answer checkpoints and raw response cache stay intact.
No new perception or memory algorithm. --check-only performs no model requests.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from benchmarks import retest_hypergraph_remaining_v5 as original
from benchmarks.comparison_worker import atomic_json
from benchmarks.findingdory_structured import fingerprint, freeze
from memory_agent.clients import ModelError
from memory_agent.settings import Settings

POLICY = {"version": "v5-json-object-recovery-v1", "max_format_retries_per_round": 2,
          "trigger": "response content cannot be parsed as a JSON object",
          "request_changes": "append JSON-format instruction only; same images/evidence/model/output limit",
          "unchanged": "retrieval, normalization, task validation, scoring and completed answers",
          "exhaustion": "stop; next manual restart gets a new persisted recovery round"}


def request_hash(url, payload, role):
    # Identical to the existing CachedClients cache key; credentials are excluded.
    return hashlib.sha256(json.dumps({"url": url, "role": role, "payload": payload},
                                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def format_error(response):
    try:
        text = response["choices"][0]["message"]["content"]
        output = json.loads(text)
        return None if isinstance(output, dict) else "json_root_is_not_object"
    except json.JSONDecodeError as exc:
        return f"invalid_json:{exc.msg}:line={exc.lineno}:column={exc.colno}"
    except (KeyError, IndexError, TypeError):
        return "missing_or_nontext_content"


def repair_payload(payload, recovery_round, attempt):
    result = deepcopy(payload)
    instruction = {"format_recovery": {
        "round": recovery_round, "attempt": attempt,
        "problem": "The previous response was not a complete parsable JSON object.",
        "instruction": "Recheck the same question and evidence. Return one complete JSON object with "
                       "goal_frames, evidence_ids and target_bindings. Close all arrays and objects; "
                       "no markdown, commentary, trailing commas or unfinished repeated lists. "
                       "Do not invent a target or evidence ID to complete the JSON."}}
    text = json.dumps(instruction, ensure_ascii=False)
    if result["messages"][-1]["role"] != "user":
        raise ValueError("Recovery expects the original answer request's user message")
    content = result["messages"][-1]["content"]
    if isinstance(content, list):
        content.append({"type": "text", "text": text})
    elif isinstance(content, str):
        result["messages"][-1]["content"] = content + "\n" + text
    else:
        raise ValueError("Unsupported answer content")
    return result


class RecoveryClients(original.previous.RetestClients):
    """Keep valid old responses; retain invalid ones and request a format retry."""
    def __init__(self, settings, budget, cache_dir, **kwargs):
        super().__init__(settings, budget, cache_dir, **kwargs)
        self.recovery_dir = self.cache_dir.parent / "json_recovery"

    def _audit(self, root, name, response, error):
        atomic_json(root / f"{name}.json", {"error": error, "response": response,
                                            "response_sha256": fingerprint(response)})

    def _trace(self, identity, recovery_round, attempt):
        self.calls.append({"role": "answer_format_recovery", "request_sha256": identity,
                           "round": recovery_round, "attempt": attempt, "policy": POLICY["version"]})

    def post(self, url, key, payload, role):
        response = super().post(url, key, payload, role)
        error = format_error(response) if role == "answer" else None
        if error is None:
            return response
        identity = request_hash(url, payload, role)
        root = self.recovery_dir / "requests" / identity
        state_path = root / "state.json"
        response_sha = fingerprint(response)
        self._audit(root, "original_invalid", response, error)
        state = original.common.load(state_path) if state_path.exists() else {
            "policy": POLICY["version"], "original_request_sha256": identity,
            "original_response_sha256": response_sha, "next_round": 0, "accepted": None}
        if (state.get("policy") != POLICY["version"] or state.get("original_request_sha256") != identity
                or state.get("original_response_sha256") != response_sha
                or type(state.get("next_round")) is not int or state["next_round"] < 0):
            raise RuntimeError("JSON recovery state differs from the preserved original response")
        accepted = state["accepted"]
        if accepted is not None:
            accepted_payload = repair_payload(payload, accepted["round"], accepted["attempt"])
            accepted_path = self.cache_dir / (request_hash(url, accepted_payload, role) + ".json")
            if not accepted_path.is_file():
                raise RuntimeError("Accepted JSON recovery cache is missing; stopped without another request")
            retried = super().post(url, key, accepted_payload, role)
            if format_error(retried) is not None or fingerprint(retried) != accepted["response_sha256"]:
                raise RuntimeError("Accepted JSON recovery response changed or disappeared")
            self._trace(identity, accepted["round"], accepted["attempt"])
            return retried
        recovery_round = state["next_round"]
        atomic_json(state_path, state)
        for attempt in range(1, POLICY["max_format_retries_per_round"] + 1):
            print(f"[JSON格式恢复] 第 {recovery_round + 1} 轮 {attempt}/2；复用同一证据和原图", flush=True)
            retried = super().post(url, key, repair_payload(payload, recovery_round, attempt), role)
            retry_error = format_error(retried)
            self._trace(identity, recovery_round, attempt)
            if retry_error is None:
                state["accepted"] = {"round": recovery_round, "attempt": attempt,
                                     "response_sha256": fingerprint(retried)}
                atomic_json(state_path, state)
                return retried
            self._audit(root, f"round_{recovery_round}_attempt_{attempt}", retried, retry_error)
        state["next_round"] = recovery_round + 1
        atomic_json(state_path, state)
        raise ModelError("answer: invalid JSON after two format retries; raw responses preserved; "
                         "same recovery command can resume with a new bounded round")


def expected_original_config(run, data, shared, source, v4, settings, manifest, all_tasks, tasks, current):
    if not (run / "config.json").is_file() or not (run / "credential_salt.bin").is_file():
        raise RuntimeError("恢复入口只用于已有且配置完整的 V5 断点目录")
    # Reconstruct the unmodified runner's configuration exactly, not a whitelist
    # that would silently permit a changed memory, model, input or answer rule.
    config = original.common.frozen_config(run, data, shared, settings, manifest, tasks)
    config.update(protocol="hypergraph-v5-remaining-errors", method="hypergraph_recall_v5", limits=original.LIMITS,
                  scope="remaining_errors_after_initial_and_v4; no full V5 accuracy", source_run=str(source), v4_run=str(v4),
                  source_config_sha256=original.common.checksum(source / "config.json"),
                  v4_config_sha256=original.common.checksum(v4 / "config.json"),
                  current_answers_sha256=fingerprint(current), selected_row_indices=[task["row_idx"] for task in tasks],
                  historical_correct_not_retested=len(all_tasks) - len(tasks))
    for name in ("retest_hypergraph_remaining_v5.py", "hypergraph_recall_v5.py", "hypergraph_visual_answer_v5.py",
                 "hypergraph_evidence_subgraph.py", "hypergraph_recall_v3.py", "hypergraph_recall_v2.py",
                 "retest_hypergraph_errors_v4.py"):
        config["source_sha256"]["benchmarks/" + name] = original.common.checksum(REPO / "benchmarks" / name)
    if original.common.load(run / "config.json") != config | {"fingerprint": fingerprint(config)}:
        raise RuntimeError("V5 原配置不匹配：模型、原代码、输入或历史断点已改变；未发模型请求")
    return config


def recovery_manifest(run):
    return {"policy": POLICY, "original_config_sha256": original.common.checksum(run / "config.json"),
            "recovery_source_sha256": original.common.checksum(Path(__file__)),
            "original_sources_unchanged": True}


def annotate_report(run):
    states = [original.common.load(path) for path in (run / "json_recovery/requests").glob("*/state.json")]
    details = {"policy": POLICY, "affected_requests": len(states),
               "recovered_requests": sum(item.get("accepted") is not None for item in states),
               "original_config_preserved": True, "usage_included_in_existing_api_ledger": True}
    atomic_json(run / "json_recovery/summary.json", details)
    summary_path = run / "summary.json"
    if summary_path.exists():
        summary = original.common.load(summary_path)
        summary["answer_format_recovery"] = details
        atomic_json(summary_path, summary)
    report_path = run / "REPORT.md"
    marker = "\n## JSON 输出恢复记录\n"
    if report_path.exists():
        report = report_path.read_text(encoding="utf-8").split(marker)[0]
        report_path.write_text(report + marker + "\n"
            f"启用额外恢复策略 `{POLICY['version']}`：仅在原响应无法解析为 JSON 对象时，"
            "使用同一模型、证据、图片和输出上限，最多追加两次格式请求；原有任务约束校验与评分保持不变。"
            f"已涉及 {details['affected_requests']} 个请求，其中 {details['recovered_requests']} 个取得可解析响应。"
            "可解析不等于答对；格式恢复的请求及 token 已纳入原账目。"
            "已完成旧答案保持原样，后续结果含此额外恢复策略，不应表述为纯原版 V5 的完整评测。\n", encoding="utf-8")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=original.common.DEFAULT_RUN)
    parser.add_argument("--v4-run", type=Path, default=original.DEFAULT_V4)
    parser.add_argument("--run-dir", type=Path, default=original.DEFAULT_RUN)
    parser.add_argument("--data-dir", type=Path, default=original.common.DEFAULT_DATA)
    parser.add_argument("--shared-perception-dir", type=Path, default=original.common.DEFAULT_SHARED)
    parser.add_argument("--max-requests", type=int, default=1_000_000)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    source, v4, run, data, shared = (path.resolve() for path in (
        args.source_run, args.v4_run, args.run_dir, args.data_dir, args.shared_perception_dir))
    paths = (source, v4, run, data, shared)
    if (args.max_requests < 1 or any(a == b or a in b.parents or b in a.parents
                                    for i, a in enumerate(paths) for b in paths[i + 1:])):
        parser.error("输入与输出必须独立，请求上限必须为正")
    settings = Settings.load(REPO / ".env")
    original.require_live_settings(settings)
    print("[恢复预检] 检查原 V5 配置、模型、共享观测及已完成题；不调用模型", flush=True)
    manifest, all_tasks, observations = original.common.verify_inputs(data, shared)
    tasks, current = original.remaining_errors(source, v4, all_tasks, shared, settings)
    expected_original_config(run, data, shared, source, v4, settings, manifest, all_tasks, tasks, current)
    answers = original.common.load_answers(run, tasks)
    if any(row.get("method") != "hypergraph_recall_v5" for row in answers.values()):
        raise RuntimeError("断点含其他方法的答案")
    pending = [task for task in tasks if task["row_idx"] not in answers]
    print(f"[断点有效] 保留已完成 {len(answers)}/{len(tasks)} 题，其中修复 {sum(row['correct'] for row in answers.values())} 题", flush=True)
    if pending:
        print(f"[下一题] {pending[0]['ep_id']} row={pending[0]['row_idx']}", flush=True)
    expected_recovery = recovery_manifest(run)
    protocol_path = run / "json_recovery/protocol.json"
    if protocol_path.exists() and original.common.load(protocol_path) != expected_recovery | {"fingerprint": fingerprint(expected_recovery)}:
        raise RuntimeError("新增恢复策略与该目录的恢复记录不一致")
    if args.check_only:
        print("[恢复预检通过] 模型请求 0；原配置和缓存未修改", flush=True)
        return
    with original.common.run_lock(run):
        freeze(protocol_path, expected_recovery)
        print("[真实模型续跑] 保留已完成答案；仅无效 JSON 追加有限次格式请求；不重新解析视频", flush=True)
        try:
            original.evaluate(run, source, shared, tasks, observations, settings, current,
                              max_requests=args.max_requests, client_factory=RecoveryClients)
        except BaseException as exc:
            atomic_json(run / "last_failure.json", {"type": type(exc).__name__})
            print(f"[中断] {type(exc).__name__}；使用同一恢复命令续跑：{run}", flush=True)
            raise
        finally:
            annotate_report(run)


if __name__ == "__main__":
    main()
