import json
import httpx
import pytest

from memory_agent.clients import ModelClients, ModelError
from memory_agent.settings import Settings


def test_qwen_non_thinking_json_and_request_limit():
    payloads = []
    def handler(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok":true}'}}]})
    client = ModelClients(Settings(llm_model="qwen3.5-27b"), httpx.MockTransport(handler), max_requests=1)
    try:
        assert client.generate("answer", "Return JSON", {}) == {"ok": True}
        assert payloads[0]["enable_thinking"] is False
        assert payloads[0]["max_tokens"] == 2048
        with pytest.raises(ModelError, match="budget exhausted"):
            client.generate("answer", "Return JSON", {})
        assert len(payloads) == 1
    finally:
        client.close()


@pytest.mark.parametrize("mismatch", [False, True])
def test_qwen_embedding_chunks_preserve_order_and_validate_across_batches(mismatch):
    sizes = []
    def handler(request):
        items = json.loads(request.content)["input"]
        sizes.append(len(items))
        vectors = [{"index": i, "embedding": [float(item) + 1, 1] +
                   ([1] if mismatch and len(sizes) > 1 else [])} for i, item in enumerate(items)]
        return httpx.Response(200, json={"data": list(reversed(vectors))})
    client = ModelClients(Settings(embedding_model="qwen3.7-text-embedding-flash"), httpx.MockTransport(handler))
    try:
        if mismatch:
            with pytest.raises(ValueError, match="dimensions differ"):
                client.embed([str(n) for n in range(21)])
        else:
            vectors = client.embed([str(n) for n in range(21)])
            assert len(vectors) == 21 and vectors[0][0] < vectors[-1][0]
        assert sizes == [20, 1]
    finally:
        client.close()
