from types import SimpleNamespace

import pytest

from app import generate
from app.pricing import cost_usd
from app.retrieve import Hit

HITS = [Hit(1, "eCFR", "https://example.test", "8 CFR 214.2(f)(10) Practical training", "90 days of unemployment", 0.9)]


class FakeClient:
    """Records the request and returns a canned message, so no API call is made."""

    last_request: dict = {}
    reply = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Ninety days [1].")],
        usage=SimpleNamespace(input_tokens=2000, output_tokens=100),
        stop_reason="end_turn",
    )

    def __init__(self, **_kwargs):
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **request):
        FakeClient.last_request = request
        return FakeClient.reply


@pytest.fixture
def fake_anthropic(monkeypatch):
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-4-5")
    monkeypatch.delenv("LLM_EFFORT", raising=False)


def test_answer_records_usage_cost_and_latency(fake_anthropic):
    result = generate.answer("How many days?", HITS)
    assert result.text == "Ninety days [1]."
    assert result.mode == "generated"
    assert (result.input_tokens, result.output_tokens) == (2000, 100)
    assert result.cost_usd == pytest.approx((2000 * 1.0 + 100 * 5.0) / 1e6)
    assert result.latency_s >= 0


def test_request_has_no_sampling_parameters(fake_anthropic):
    generate.answer("How many days?", HITS)
    sent = generate_request = FakeClient.last_request
    for removed in ("temperature", "top_p", "top_k"):
        assert removed not in sent  # rejected by the API from 2026-10-21 and by anthropic>=1 types
    assert "output_config" not in generate_request


def test_effort_is_sent_only_when_configured(fake_anthropic, monkeypatch):
    monkeypatch.setenv("LLM_EFFORT", "low")
    generate.answer("How many days?", HITS)
    assert FakeClient.last_request["output_config"] == {"effort": "low"}


def test_system_prompt_pins_the_abstention_sentence(fake_anthropic):
    generate.answer("How many days?", HITS)
    assert generate.ABSTAIN_PHRASE in FakeClient.last_request["system"]


def test_refusal_and_truncation_are_not_returned_as_answers(fake_anthropic):
    FakeClient.reply = SimpleNamespace(content=[], usage=SimpleNamespace(input_tokens=5, output_tokens=0), stop_reason="refusal")
    assert generate.answer("q", HITS).mode == "refused"
    FakeClient.reply = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="cut off")], usage=SimpleNamespace(input_tokens=5, output_tokens=4000), stop_reason="max_tokens"
    )
    assert generate.answer("q", HITS).mode == "truncated"


def test_without_a_key_generation_is_skipped(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert generate.answer("q", HITS).mode == "retrieval-only"


def test_unknown_model_has_no_cost_rather_than_a_guess():
    assert cost_usd("claude-not-a-real-model", 1000, 1000) is None
