"""Offline contracts for direct TypeSafe and Vercel AI Gateway requests."""

from jev_ultrafast import model

BODY = {
    "model": "typesafe-ai/jev",
    "state": {"page": {}},
    "questions": {"operation": {"type": "choice"}},
}


def test_direct_typesafe_request_is_unchanged(monkeypatch):
    calls = []
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    monkeypatch.setattr(
        model,
        "post_json",
        lambda url, key, body, headers=None: calls.append((url, key, body, headers)) or {"answers": {}},
    )

    model.typesafe_request(BODY)

    assert calls == [("https://api.typesafe.ai/v1/systemone", "test-key", BODY, None)]


def test_vercel_gateway_request_maps_protocol_and_normalizes_probabilities(monkeypatch):
    calls = []
    monkeypatch.setenv("TYPESAFE_API_KEY", "gateway-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://ai-gateway.vercel.sh/")
    reply = {
        "answers": {
            "operation": {
                "type": "choice",
                "choice": "CLICK",
                "probabilities": {"CLICK": 0.67, "DONE": 0.34},
            }
        },
        "usage": {"inputTokens": 12, "outputTokens": 3},
        "providerMetadata": {"typesafe": {"confidence": {"operation": 0.91}}},
    }
    monkeypatch.setattr(
        model,
        "post_json",
        lambda url, key, body, headers=None: calls.append((url, key, body, headers)) or reply,
    )

    result = model.typesafe_request(BODY)

    url, key, body, headers = calls[0]
    assert url == "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
    assert key == "gateway-key"
    assert body == {"state": BODY["state"], "questions": BODY["questions"]}
    assert headers["ai-model-id"] == "typesafe-ai/jev"
    assert headers["ai-evaluation-model-specification-version"] == "4"
    answer = result["answers"]["operation"]
    assert abs(sum(answer["probabilities"].values()) - 1) < 1e-9
    assert answer["choice"] == "CLICK"
    assert answer["confidence"] == 0.91
    assert result["usage"] == {"input_tokens": 12, "output_tokens": 3}
