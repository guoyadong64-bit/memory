"""Small live connectivity checks using synthetic data, never stored as memory."""
from .clients import ModelClients, ModelError
from .models import Query, Record
from .settings import Settings


def probe(settings: Settings, clients: ModelClients | None = None) -> dict:
    settings.validate_live()
    config = settings.model_copy(update={"mode": "live", "retries": 0,
                                         "timeout": min(settings.timeout, 20)})
    client = clients or ModelClients(config)
    results = []
    try:
        roles = ["answer"]
        if ((settings.understanding_model or settings.llm_model)
                != (settings.answer_model or settings.llm_model)):
            roles.append("understanding")
        for role in roles:
            try:
                output = client.generate(role, 'Return exactly the JSON object {"ok":true}.',
                                         {"purpose": "synthetic connection test"})
                if output.get("ok") is not True:
                    raise ModelError("generation: unexpected JSON test answer")
                results.append({"service": "llm_" + role, "ok": True})
            except ModelError as error:
                results.append({"service": "llm_" + role, "ok": False, "error": str(error)})
        record = Record(id="connection_test", key="demo.color", question="What color is the sample?",
                        answer="Blue", source="synthetic test data", verified=True)
        try:
            decision = client.judge(Query(question=record.question, key=record.key), [record])
            results.append({"service": "jev", "ok": True, "model": decision.model,
                            "action": decision.action})
        except ModelError as error:
            results.append({"service": "jev", "ok": False, "error": str(error)})
        return {"ok": all(r["ok"] for r in results), "checks": results,
                "calls": list(client.calls), "memory_written": False}
    finally:
        if clients is None:
            client.close()
