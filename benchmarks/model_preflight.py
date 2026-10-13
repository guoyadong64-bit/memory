"""Four-request, no-retry connectivity and visual schema check; never a benchmark score."""
from pathlib import Path
import json

from memory_agent.clients import ModelClients, ModelError
from memory_agent.probe import probe
from memory_agent.settings import Settings
from benchmarks.findingdory_structured import extract_batch
from benchmarks.findingdory_pilot import save


def main():
    settings = Settings.load().model_copy(update={"mode": "live", "retries": 0, "timeout": 45})
    settings.validate_live()
    out = Path("runtime/model_check_20261012")
    client = ModelClients(settings, max_requests=4)
    checks = []
    try:
        # Current answer/understanding are the same model; no repeated text requests.
        basic = probe(settings, client)
        checks.extend(basic["checks"])
        try:
            vectors = client.embed(["The lamp is on the cabinet.", "台灯放在柜子上。"])
            checks.append({"service": "embedding", "ok": True, "vectors": len(vectors),
                           "dimension": len(vectors[0])})
        except (ModelError, ValueError) as exc:
            checks.append({"service": "embedding", "ok": False, "error": str(exc)})
        if next((c["ok"] for c in checks if c["service"] == "llm_answer"), False):
            try:
                frames = extract_batch(client, Path("runtime/findingdory"), "ep_1", [2],
                    diagnostic_path=out / "vision_errors.json", allow_split=False,
                    max_attempts=1, max_output_tokens=1200)
                save(out / "vision_observations.json", [f.model_dump(mode="json") for f in frames])
                checks.append({"service": "vision", "ok": True, "accepted_frames": 1,
                               "scope": "schema/identity validation, not independent visual truth"})
            except (ModelError, RuntimeError) as exc:
                checks.append({"service": "vision", "ok": False, "error": str(exc)})
        else:
            checks.append({"service": "vision", "ok": False, "error": "Skipped: text authentication failed"})
    finally:
        report = {"not_a_benchmark_score": True, "request_limit": 4,
            "requests_sent": client.requests_sent, "automatic_retries": 0,
            "models": {"llm": settings.llm_model, "vision": settings.vision_model or settings.llm_model,
                       "embedding": settings.embedding_model, "jev": settings.jev_model},
            "thinking_enabled": settings.llm_thinking, "checks": checks, "calls": client.calls,
            "ok": bool(checks) and all(c["ok"] for c in checks), "memory_written": False,
            "reported_prompt_tokens": sum(c.get("usage", {}).get("prompt_tokens", 0) for c in client.calls),
            "reported_completion_tokens": sum(c.get("usage", {}).get("completion_tokens", 0) for c in client.calls),
            "reported_total_tokens_all_services": sum(c.get("usage", {}).get("total_tokens",
                c.get("usage", {}).get("input_tokens", 0) + c.get("usage", {}).get("output_tokens", 0))
                for c in client.calls)}
        save(out / "report.json", report)
        client.close()
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
