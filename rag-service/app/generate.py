import time
from dataclasses import dataclass

from app.config import load_settings
from app.pricing import cost_usd
from app.retrieve import Hit
from app.tracing import observe

DISCLAIMER = (
    "This is general information from public regulations, not legal advice. "
    "Confirm anything important with your school's DSO or an immigration attorney."
)

# The exact sentence the model must use when the sources do not answer. The answer eval and the
# abstention metric match on this string, so changing it changes what counts as an abstention.
ABSTAIN_PHRASE = "The indexed sources do not answer this."

SYSTEM_PROMPT = f"""You answer questions about F-1 student status, OPT and STEM OPT using ONLY the numbered sources provided.
Rules:
- Every factual sentence must end with the source number(s) in square brackets, like [1] or [2][3].
- If the sources do not contain the answer, reply with exactly: "{ABSTAIN_PHRASE}" and nothing else. Do not use outside knowledge, and do not guess.
- Quote numbers and deadlines exactly as the sources state them.
- Be concise. Do not give legal advice or tell the user what they should do in their own case."""

# Thinking models spend part of max_tokens on reasoning, so leave generous headroom.
MAX_TOKENS = 4000


@dataclass
class Answer:
    text: str | None
    mode: str  # "generated" | "retrieval-only" | "refused" | "truncated"
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float | None = None
    stop_reason: str | None = None


def format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] {h.citation}\n{h.content}" for i, h in enumerate(hits, start=1))


def answer(question: str, hits: list[Hit]) -> Answer:
    settings = load_settings()
    if not settings.anthropic_api_key or not hits:
        return Answer(None, "retrieval-only")
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    request: dict = {
        "model": settings.llm_model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"}
        ],
    }
    if settings.llm_effort:
        request["output_config"] = {"effort": settings.llm_effort}
    with observe(
        "generate",
        as_type="generation",
        model=settings.llm_model,
        model_parameters={"max_tokens": MAX_TOKENS, "effort": settings.llm_effort or "default"},
        input={"question": question, "passages": [h.citation for h in hits]},
    ) as span:
        started = time.perf_counter()
        message = client.messages.create(**request)
        latency = time.perf_counter() - started
        text = "".join(b.text for b in message.content if b.type == "text") or None
        mode = "generated"
        if message.stop_reason == "refusal":
            mode, text = "refused", None
        elif message.stop_reason == "max_tokens":
            mode = "truncated"
        cost = cost_usd(settings.llm_model, message.usage.input_tokens, message.usage.output_tokens)
        details: dict = {
            "output": text,
            "usage_details": {"input": message.usage.input_tokens, "output": message.usage.output_tokens},
            "metadata": {"stop_reason": message.stop_reason, "mode": mode},
        }
        if cost is not None:
            details["cost_details"] = {"total": cost}
        span.update(**details)
    return Answer(
        text=text,
        mode=mode,
        model=settings.llm_model,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        latency_s=latency,
        cost_usd=cost,
        stop_reason=message.stop_reason,
    )
