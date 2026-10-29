"""LLM-as-judge for claim support. The judge only decides support; scoring lives in answer_metrics."""
import json
import time
from dataclasses import dataclass
from typing import Callable

from app.config import load_settings
from app.pricing import cost_usd
from eval.answer_metrics import Claim, Verdict

JUDGE_MAX_TOKENS = 4000

JUDGE_SYSTEM = """You are a strict fact-checker for answers about US student immigration rules.
You are given numbered source passages, a question, and the claims (numbered from 1) extracted from an answer, each with the passage numbers it cited.
For every claim decide two things, using ONLY the passages (never your own knowledge):
- supported_by_any: the claim is stated or directly entailed by at least one of the passages. Numbers, durations and conditions must match exactly. A claim that adds a detail, a number, or a condition the passages do not contain is NOT supported.
- supported_by_cited: the claim is stated or directly entailed by the passages whose numbers the claim cites. If the claim cites nothing, or cites a passage that does not contain it, this is false, even when another passage supports it.
Be literal. When unsure, answer false. Give a one-sentence reason."""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "supported_by_any": {"type": "boolean"},
                    "supported_by_cited": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "supported_by_any", "supported_by_cited", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

JudgeFn = Callable[[str, list[Claim], list[str]], list[Verdict]]


@dataclass
class JudgeUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0
    latency_s: float = 0.0


def build_prompt(question: str, claims: list[Claim], passages: list[str]) -> str:
    sources = "\n\n".join(f"[{i}]\n{p}" for i, p in enumerate(passages, start=1))
    listed = "\n".join(
        f"{i}. {c.text}  (cites: {', '.join(map(str, c.cited)) if c.cited else 'nothing'})"
        for i, c in enumerate(claims, start=1)
    )
    return f"Passages:\n\n{sources}\n\nQuestion: {question}\n\nClaims:\n{listed}"


def parse_verdicts(raw_json: str, n_claims: int) -> list[Verdict]:
    """Map the judge's JSON onto claims. A claim the judge skipped or garbled is unsupported, never passed."""
    verdicts = [Verdict(False, False, "judge returned no verdict for this claim") for _ in range(n_claims)]
    try:
        items = json.loads(raw_json)["claims"]
    except (ValueError, KeyError, TypeError):
        return verdicts
    for item in items:
        i = item.get("index") if isinstance(item, dict) else None
        if isinstance(i, int) and 1 <= i <= n_claims:
            verdicts[i - 1] = Verdict(
                bool(item.get("supported_by_cited")), bool(item.get("supported_by_any")), str(item.get("reason", ""))
            )
    return verdicts


def make_llm_judge(usage: JudgeUsage) -> JudgeFn:
    import anthropic

    settings = load_settings()
    if not settings.judge_model:
        raise RuntimeError("JUDGE_MODEL is not set")
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def judge(question: str, claims: list[Claim], passages: list[str]) -> list[Verdict]:
        request: dict = {
            "model": settings.judge_model,
            "max_tokens": JUDGE_MAX_TOKENS,
            "system": JUDGE_SYSTEM,
            "messages": [{"role": "user", "content": build_prompt(question, claims, passages)}],
            "output_config": {"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
        }
        started = time.perf_counter()
        message = client.messages.create(**request)
        usage.latency_s += time.perf_counter() - started
        usage.calls += 1
        usage.input_tokens += message.usage.input_tokens
        usage.output_tokens += message.usage.output_tokens
        call_cost = cost_usd(settings.judge_model, message.usage.input_tokens, message.usage.output_tokens)
        usage.cost_usd = None if call_cost is None or usage.cost_usd is None else usage.cost_usd + call_cost
        text = next((b.text for b in message.content if b.type == "text"), "")
        return parse_verdicts(text, len(claims))

    return judge
