import re
from dataclasses import dataclass

import psycopg

from app.embeddings import embed_query
from app.rerank import rerank_scores
from app.tracing import observe

RRF_K = 60
CANDIDATES = 30
RERANK_POOL = 20
# The regulation is the authority; Federal Register preamble is explanatory commentary and is far
# larger, so unweighted fusion lets commentary crowd out the rule text. Commentary still surfaces
# when it is clearly the better match.
SOURCE_WEIGHT = {"eCFR": 1.0, "Saved page": 1.0, "Federal Register": 0.6}
LEG_WEIGHT = {"dense": 1.0, "keyword": 0.5}
# The regulation spells these out and the embedding model does not reliably link an abbreviation to its
# phrase: for "Does OPT authorization end if I transfer to another school?" the chunk that answers it
# ("optional practical training ... terminated when the student transfers") ranked 146th of 550.
GLOSSARY = {
    "pdso": "principal designated school official",
    "dso": "designated school official",
    "opt": "optional practical training",
    "cpt": "curricular practical training",
    "sevis": "Student and Exchange Visitor Information System",
    "sevp": "Student and Exchange Visitor Program",
    "ead": "employment authorization document",
}
_ABBREVIATION = re.compile(r"\b(" + "|".join(GLOSSARY) + r")\b", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "can", "i", "my", "me", "we",
    "to", "of", "in", "on", "for", "and", "or", "if", "how", "what", "when", "which", "who", "it",
    "be", "am", "with", "that", "this", "at", "by", "as", "from", "have", "has", "will", "may",
}


@dataclass
class Hit:
    id: int
    source: str
    source_url: str
    citation: str
    content: str
    score: float


def expand_query(question: str) -> str:
    """Append the full phrase after the first use of each known abbreviation, keeping the user's words."""
    seen: set[str] = set()

    def spell_out(match: re.Match) -> str:
        key = match.group(1).lower()
        if key in seen or GLOSSARY[key] in question.lower():
            return match.group(0)
        seen.add(key)
        return f"{match.group(0)} ({GLOSSARY[key]})"

    return _ABBREVIATION.sub(spell_out, question)


def _or_query(question: str) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in STOPWORDS and len(w) > 1]
    return " | ".join(dict.fromkeys(words))


def _dense(conn: psycopg.Connection, question: str, limit: int) -> list[int]:
    vec = embed_query(question)
    rows = conn.execute("SELECT id FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s", (vec, limit)).fetchall()
    return [r[0] for r in rows]


def _keyword(conn: psycopg.Connection, question: str, limit: int) -> list[int]:
    query = _or_query(question)
    if not query:
        return []
    rows = conn.execute(
        "SELECT id FROM chunks WHERE tsv @@ to_tsquery('english', %s) "
        "ORDER BY ts_rank_cd(tsv, to_tsquery('english', %s)) DESC LIMIT %s",
        (query, query, limit),
    ).fetchall()
    return [r[0] for r in rows]


def search(conn: psycopg.Connection, question: str, top_k: int = 6, mode: str = "hybrid") -> list[Hit]:
    expanded = expand_query(question)
    with observe(
        "retrieve", as_type="retriever", input={"question": question, "mode": mode, "top_k": top_k}, metadata={"expanded_query": expanded}
    ) as span:
        hits = _search(conn, expanded, top_k, mode, rerank_question=question)
        span.update(output=[{"citation": h.citation, "score": round(h.score, 4)} for h in hits])
    return hits


def _search(conn: psycopg.Connection, question: str, top_k: int = 6, mode: str = "hybrid", rerank_question: str | None = None) -> list[Hit]:
    """The candidate legs search `question` as given (callers pass the abbreviation-expanded text), while the cross-encoder
    scores against `rerank_question`, the user's own words: expanding everywhere measured worse on both question sets.

    mode: 'dense' (vector only), 'keyword' (full text only), 'hybrid' (reciprocal rank fusion)
    or 'rerank' (hybrid pool re-scored by a cross-encoder)."""
    rankings: list[tuple[str, list[int]]] = []
    if mode in ("dense", "hybrid", "rerank"):
        rankings.append(("dense", _dense(conn, question, CANDIDATES)))
    if mode in ("keyword", "hybrid", "rerank"):
        rankings.append(("keyword", _keyword(conn, question, CANDIDATES)))
    candidates = {i for _, ranking in rankings for i in ranking}
    if not candidates:
        return []
    rows = {
        r[0]: r
        for r in conn.execute(
            "SELECT id, source, source_url, citation, content FROM chunks WHERE id = ANY(%s)", (list(candidates),)
        )
    }
    scores: dict[int, float] = {}
    for leg, ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            weight = LEG_WEIGHT[leg] if mode in ("hybrid", "rerank") else 1.0
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (RRF_K + rank + 1)
    for chunk_id in scores:
        scores[chunk_id] *= SOURCE_WEIGHT.get(rows[chunk_id][1], 1.0)
    if mode == "rerank":
        pool = sorted(scores, key=scores.get, reverse=True)[:RERANK_POOL]
        with observe("rerank", input={"candidates": len(pool)}):
            relevance = rerank_scores(rerank_question or question, [rows[i][4] for i in pool])
        scores = {i: p * SOURCE_WEIGHT.get(rows[i][1], 1.0) for i, p in zip(pool, relevance)}
    best = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [Hit(*rows[i], score=scores[i]) for i in best]
