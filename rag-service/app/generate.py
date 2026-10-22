from dataclasses import dataclass

from app.config import load_settings
from app.retrieve import Hit

DISCLAIMER = (
    "This is general information from public regulations, not legal advice. "
    "Confirm anything important with your school's DSO or an immigration attorney."
)

SYSTEM_PROMPT = """You answer questions about F-1 student status, OPT and STEM OPT using ONLY the numbered sources provided.
Rules:
- Every factual sentence must end with the source number(s) in square brackets, like [1] or [2][3].
- If the sources do not contain the answer, say exactly: "The indexed sources do not answer this." Do not use outside knowledge.
- Quote numbers and deadlines exactly as the sources state them.
- Be concise. Do not give legal advice or tell the user what they should do in their own case."""


@dataclass
class Answer:
    text: str | None
    mode: str


def _format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] {h.citation}\n{h.content}" for i, h in enumerate(hits, start=1))


def answer(question: str, hits: list[Hit]) -> Answer:
    settings = load_settings()
    if not settings.anthropic_api_key or not hits:
        return Answer(None, "retrieval-only")
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    message = client.messages.create(
        model=settings.llm_model,
        max_tokens=700,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Sources:\n\n{_format_sources(hits)}\n\nQuestion: {question}"}],
    )
    return Answer("".join(b.text for b in message.content if b.type == "text"), "generated")
