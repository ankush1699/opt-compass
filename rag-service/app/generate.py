import time
from collections.abc import Iterator
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
- Each source is labelled REGULATION or COMMENTARY. COMMENTARY gives the government's reasoning and answers "why" questions; use it for that. For what the rule currently requires, a REGULATION source wins: if the two disagree, state only what the REGULATION says.
- Be concise. Do not give legal advice or tell the user what they should do in their own case."""

# What each retrieved source is, shown in the prompt. The Federal Register text is the 2016 preamble: it explains
# the rule as published and quotes earlier and proposed text, so where it disagrees with the current regulation
# the regulation wins (q025: "April 1" in the rule against "October 1" in the commentary).
SOURCE_LABEL = {
    "eCFR": "REGULATION (current law)",
    "Federal Register": "COMMENTARY (2016 preamble: the government's reasoning; it can quote earlier or proposed text)",
}

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
    return "\n\n".join(
        f"[{i}] {SOURCE_LABEL.get(h.source, h.source)}: {h.citation}\n{h.content}" for i, h in enumerate(hits, start=1)
    )


def _provider_for(hits: list[Hit], provider):
    """The provider to use, or None for retrieval-only mode (no hits, or no credentials for the configured provider)."""
    settings = load_settings()
    if not hits:
        return None
    if provider is not None:
        return provider
    if not has_credentials(settings.llm_provider, settings):
        return None
    return make_provider(settings.llm_provider, settings, settings.llm_model, settings.llm_effort)


def _generation_span(question: str, hits: list[Hit]):
    settings = load_settings()
    return observe(
        "generate",
        as_type="generation",
        model=settings.llm_model,
        model_parameters={"provider": settings.llm_provider, "max_tokens": MAX_TOKENS, "effort": settings.llm_effort or "default"},
        input={"question": question, "passages": [h.citation for h in hits]},
    )


def _finish(result, latency: float, span) -> Answer:
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
        model=load_settings().llm_model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_s=latency,
        cost_usd=result.cost_usd,
        stop_reason=result.stop_reason,
    )


def _prompt(question: str, hits: list[Hit]) -> str:
    return f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"


def answer(question: str, hits: list[Hit], provider=None) -> Answer:
    """Generate a cited answer. `provider` is injectable for tests; by default it comes from settings."""
    provider = _provider_for(hits, provider)
    if provider is None:
        return Answer(None, "retrieval-only")
    with _generation_span(question, hits) as span:
        started = time.perf_counter()
        result = provider.complete(SYSTEM_PROMPT, _prompt(question, hits), MAX_TOKENS)
        return _finish(result, time.perf_counter() - started, span)


def answer_stream(question: str, hits: list[Hit], provider=None) -> Iterator[str | Answer]:
    """Like `answer`, but yields text deltas as the model produces them and ends with the final Answer.

    Drive it from one thread: the tracing span stays open across the yields.
    """
    provider = _provider_for(hits, provider)
    if provider is None:
        yield Answer(None, "retrieval-only")
        return
    with _generation_span(question, hits) as span:
        started = time.perf_counter()
        result = None
        for item in provider.stream(SYSTEM_PROMPT, _prompt(question, hits), MAX_TOKENS):
            if isinstance(item, str):
                yield item
            else:
                result = item
        yield _finish(result, time.perf_counter() - started, span)
