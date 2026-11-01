import json

import httpx
import pytest

from app.config import load_settings
from app.llm import GeminiProvider, ProviderError, _strip_additional_properties


def provider(handler, monkeypatch, free="1", sleeps=None):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_FREE_TIER", free)
    monkeypatch.setenv("GEMINI_MIN_INTERVAL_S", "0")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GeminiProvider(load_settings(), "gemini-test-model", client=client, sleep=(sleeps.append if sleeps is not None else (lambda s: None)))


def ok(text="Ninety days [1].", prompt=1200, candidates=80, thoughts=40, finish="STOP"):
    return httpx.Response(200, json={
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}],
        "usageMetadata": {"promptTokenCount": prompt, "candidatesTokenCount": candidates, "thoughtsTokenCount": thoughts},
    })


def test_request_shape_keeps_the_key_out_of_the_url_and_has_no_stray_fields(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"], seen["headers"], seen["body"] = str(request.url), request.headers, json.loads(request.content)
        return ok()

    provider(handler, monkeypatch).complete("SYSTEM", "USER", 500)
    assert "key" not in seen["url"].split("?")[-1] and "test-key-not-real" not in seen["url"]
    assert seen["headers"]["x-goog-api-key"] == "test-key-not-real"
    assert seen["url"].endswith("models/gemini-test-model:generateContent")
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "SYSTEM"
    assert seen["body"]["contents"][0]["parts"][0]["text"] == "USER"
    assert seen["body"]["generationConfig"] == {"maxOutputTokens": 500, "temperature": 0}


def test_json_schema_is_sent_without_additional_properties(monkeypatch):
    seen = {}

    def handler(request):
        seen["config"] = json.loads(request.content)["generationConfig"]
        return ok("{}")

    schema = {"type": "object", "properties": {"a": {"type": "object", "additionalProperties": False}}, "additionalProperties": False}
    provider(handler, monkeypatch).complete("s", "u", 100, schema)
    assert seen["config"]["responseMimeType"] == "application/json"
    assert "additionalProperties" not in json.dumps(seen["config"]["responseSchema"])
    assert "additionalProperties" in json.dumps(schema)  # the caller's schema is not mutated
    assert _strip_additional_properties({"x": [{"additionalProperties": 1, "y": 2}]}) == {"x": [{"y": 2}]}


def test_thinking_tokens_count_as_output_and_free_tier_costs_nothing(monkeypatch):
    result = provider(lambda r: ok(), monkeypatch).complete("s", "u", 100)
    assert (result.input_tokens, result.output_tokens) == (1200, 120)
    assert result.cost_usd == 0.0 and result.stop_reason == "end_turn"


def test_paid_tier_does_not_pretend_to_know_the_cost(monkeypatch):
    assert provider(lambda r: ok(), monkeypatch, free="0").complete("s", "u", 100).cost_usd is None


def test_retries_rate_limits_using_the_servers_retry_delay(monkeypatch):
    calls, sleeps = [], []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, text='{"error": {"details": [{"retryDelay": "12s"}]}}')
        return ok()

    result = provider(handler, monkeypatch, sleeps=sleeps).complete("s", "u", 100)
    assert len(calls) == 2 and result.text == "Ninety days [1]."
    assert 13.0 in sleeps  # 12s from the server plus one second of margin


def test_gives_up_after_repeated_rate_limits_instead_of_returning_garbage(monkeypatch):
    with pytest.raises(ProviderError):
        provider(lambda r: httpx.Response(429, text="slow down"), monkeypatch).complete("s", "u", 100)


def test_client_errors_are_raised_not_retried(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, text="bad request")

    with pytest.raises(ProviderError, match="400"):
        provider(handler, monkeypatch).complete("s", "u", 100)
    assert len(calls) == 1


def test_blocked_and_truncated_responses_are_flagged(monkeypatch):
    assert provider(lambda r: ok(text="x", finish="SAFETY"), monkeypatch).complete("s", "u", 1).stop_reason == "refusal"
    assert provider(lambda r: httpx.Response(200, json={"candidates": [], "usageMetadata": {}}), monkeypatch).complete("s", "u", 1).text is None
    assert provider(lambda r: ok(finish="MAX_TOKENS"), monkeypatch).complete("s", "u", 1).stop_reason == "max_tokens"


def test_refuses_to_sleep_for_hours_when_the_daily_quota_is_exhausted(monkeypatch):
    sleeps = []
    quota = httpx.Response(429, text='{"error": {"details": [{"retryDelay": "9900s"}]}}')
    with pytest.raises(ProviderError, match="quota exhausted"):
        provider(lambda r: quota, monkeypatch, sleeps=sleeps).complete("s", "u", 100)
    assert sleeps == []


def sse(*chunks):
    body = "".join(f"data: {json.dumps(c)}\r\n\r\n" for c in chunks)
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def test_stream_yields_text_deltas_then_one_result_with_usage(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        return sse(
            {"candidates": [{"content": {"parts": [{"text": "Ninety "}]}}]},
            {"candidates": [{"content": {"parts": [{"text": "days [1]."}]}, "finishReason": "STOP"}],
             "usageMetadata": {"promptTokenCount": 1200, "candidatesTokenCount": 6, "thoughtsTokenCount": 4}},
        )

    items = list(provider(handler, monkeypatch).stream("SYSTEM", "USER", 500))
    assert items[:2] == ["Ninety ", "days [1]."]
    result = items[2]
    assert (result.text, result.stop_reason) == ("Ninety days [1].", "end_turn")
    assert (result.input_tokens, result.output_tokens, result.cost_usd) == (1200, 10, 0.0)
    assert seen["url"].endswith(":streamGenerateContent?alt=sse")
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "SYSTEM"


def test_stream_retries_before_the_first_byte_and_skips_thought_parts(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, text="high demand")
        return sse({"candidates": [{"content": {"parts": [{"text": "hidden", "thought": True}, {"text": "Yes [1]."}]}, "finishReason": "STOP"}]})

    sleeps = []
    items = list(provider(handler, monkeypatch, sleeps=sleeps).stream("S", "U", 100))
    assert len(calls) == 2 and len(sleeps) == 1
    assert items[0] == "Yes [1]." and items[-1].text == "Yes [1]."


def test_stream_raises_on_a_client_error_and_on_quota_exhaustion(monkeypatch):
    with pytest.raises(ProviderError, match="400"):
        list(provider(lambda r: httpx.Response(400, text="bad"), monkeypatch).stream("S", "U", 100))
    quota = httpx.Response(429, text='{"error": {"details": [{"retryDelay": "5000s"}]}}')
    with pytest.raises(ProviderError, match="quota exhausted"):
        list(provider(lambda r: quota, monkeypatch).stream("S", "U", 100))


def test_timeouts_and_dropped_connections_become_provider_errors(monkeypatch):
    def hang(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderError, match="ReadTimeout"):
        provider(hang, monkeypatch).complete("S", "U", 100)
    with pytest.raises(ProviderError, match="ReadTimeout"):
        list(provider(hang, monkeypatch).stream("S", "U", 100))


def test_make_provider_refuses_an_empty_model_name():
    from app.config import load_settings
    from app.llm import ProviderError, make_provider

    for model in ("", "  "):
        with pytest.raises(ProviderError, match="LLM_MODEL"):
            make_provider("gemini", load_settings(), model)
