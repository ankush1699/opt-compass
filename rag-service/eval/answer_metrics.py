"""Pure scoring logic for the answer-quality eval: no network, no model calls.

The judge (an LLM, or a stub in tests) decides whether a claim is supported. Everything else,
splitting an answer into claims, reading the [n] citations, detecting abstention, aggregating and
gating, is deterministic so it can be unit-tested and so a wrong answer provably fails.
"""
import re
from dataclasses import dataclass, field

from app.generate import ABSTAIN_PHRASE

_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?\]])\s+(?=[A-Z0-9(\"“])")


@dataclass
class Claim:
    text: str
    cited: list[int]


@dataclass
class Verdict:
    supported_by_cited: bool
    supported_by_any: bool
    reason: str = ""


@dataclass
class ScoredAnswer:
    kind: str  # "answerable" | "unanswerable"
    abstained: bool
    claims: list[Claim] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
    n_passages: int = 6


def is_abstention(text: str | None) -> bool:
    return bool(text) and ABSTAIN_PHRASE.lower() in text.lower()


def split_claims(text: str) -> list[Claim]:
    """One claim per sentence. Citation markers are removed from the text and returned as numbers."""
    claims: list[Claim] = []
    for line in text.splitlines():
        for sentence in _SENTENCE_BREAK.split(line.strip()):
            cited = [int(n) for m in _MARKER.findall(sentence) for n in re.split(r"\s*,\s*", m)]
            clean = _MARKER.sub("", sentence).strip(" \t-*•")
            if len(clean) >= 3:
                claims.append(Claim(clean, sorted(set(cited))))
    return claims


def valid_citations(claim: Claim, n_passages: int) -> list[int]:
    return [n for n in claim.cited if 1 <= n <= n_passages]


def aggregate(items: list[ScoredAnswer]) -> dict:
    answerable = [i for i in items if i.kind == "answerable"]
    unanswerable = [i for i in items if i.kind == "unanswerable"]
    answered = [i for i in answerable if not i.abstained]

    claims = [(c, v, i.n_passages) for i in answered for c, v in zip(i.claims, i.verdicts)]
    cited = [(c, v, n) for c, v, n in claims if valid_citations(c, n)]
    metrics: dict = {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "n_claims": len(claims),
        "faithfulness": _ratio(sum(v.supported_by_any for _, v, _ in claims), len(claims)),
        "citation_precision": _ratio(sum(v.supported_by_cited for _, v, _ in cited), len(cited)),
        "citation_coverage": _ratio(len(cited), len(claims)),
        "invalid_citation_rate": _ratio(
            sum(1 for c, _, n in claims if any(not 1 <= x <= n for x in c.cited)), len(claims)
        ),
        "fully_faithful_answers": _ratio(
            sum(1 for i in answered if i.claims and all(v.supported_by_any for v in i.verdicts)), len(answered)
        ),
        "false_abstain_rate": _ratio(sum(i.abstained for i in answerable), len(answerable)),
        "abstain_rate_on_unanswerable": _ratio(sum(i.abstained for i in unanswerable), len(unanswerable)),
    }
    return metrics


def _ratio(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


DEFAULT_THRESHOLDS = {
    "faithfulness": ("min", 0.90),
    "citation_precision": ("min", 0.85),
    "abstain_rate_on_unanswerable": ("min", 0.80),
    "false_abstain_rate": ("max", 0.15),
}


def gate(metrics: dict, thresholds: dict = DEFAULT_THRESHOLDS) -> list[str]:
    """Return a list of failures; empty means the answers pass. A metric with no data fails."""
    failures = []
    for name, (direction, limit) in thresholds.items():
        value = metrics.get(name)
        if value is None:
            failures.append(f"{name}: no data")
        elif direction == "min" and value < limit:
            failures.append(f"{name}: {value:.2f} < {limit:.2f}")
        elif direction == "max" and value > limit:
            failures.append(f"{name}: {value:.2f} > {limit:.2f}")
    return failures


def cohens_kappa(a: list[bool], b: list[bool]) -> float | None:
    """Agreement between two binary raters corrected for chance; None when undefined."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)
