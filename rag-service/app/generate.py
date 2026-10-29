import time
from dataclasses import dataclass

from app.config import load_settings
from app.llm import has_credentials, make_provider
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


def answer(question: str, hits: list[Hit], provider=None) -> Answer:
    """Generate a cited answer. `provider` is injectable for tests; by default it comes from settings."""
    settings = load_settings()
    if provider is None:
        if not has_credentials(settings.llm_provider, settings) or not hits:
            return Answer(None, "retrieval-only")
        provider = make_provider(settings.llm_provider, settings, settings.llm_model, settings.llm_effort)
    elif not hits:
        return Answer(None, "retrieval-only")
    user = f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"
    with observe(
        "generate",
        as_type="generation",
        model=settings.llm_model,
        model_parameters={"provider": settings.llm_provider, "max_tokens": MAX_TOKENS, "effort": settings.llm_effort or "default"},
        input={"question": question, "passages": [h.citation for h in hits]},
    ) as span:
        started = time.perf_counter()
        result = provider.complete(SYSTEM_PROMPT, user, MAX_TOKENS)
        latency = time.perf_counter() - started
        mode = {"refusal": "refused", "max_tokens": "truncated"}.get(result.stop_reason or "", "generated")
        details: dict = {
            "output": result.text,
            "usage_details": {"input": result.input_tokens, "output": result.output_tokens},
            "metadata": {"stop_reason": result.stop_reason, "mode": mode},
        }
        if result.cost_usd is not None:
            details["cost_details"] = {"total": result.cost_usd}
        span.update(**details)
    return Answer(
        text=result.text,
        mode=mode,
        model=settings.llm_model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_s=latency,
        cost_usd=result.cost_usd,
        stop_reason=result.stop_reason,
    )
