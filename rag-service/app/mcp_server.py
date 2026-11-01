"""MCP server: lets Claude Desktop, Claude Code or any MCP client search and quote the indexed regulations.

Run over stdio (the client starts it):  python -m app.mcp_server
Tools: search_regulations(query, top_k), get_section(citation_id), ask(question).

Over stdio the protocol uses stdout, so nothing here may print to it; diagnostics go to stderr.
"""
import re

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError  # its message reaches the client; other exceptions are hidden as "unexpected"
from psycopg.pq import TransactionStatus
from pydantic import BaseModel

from app.db import connect
from app.generate import DISCLAIMER, answer
from app.llm import ProviderError
from app.retrieve import search

MAX_TOP_K = 15
MAX_SECTION_CHARS = 20_000

mcp = MCPServer(
    "opt-compass",
    instructions=(
        "Search and quote the text of 8 CFR 214.2(f) (F-1 students, OPT, STEM OPT; current eCFR) and the 2016 STEM OPT "
        "final rule preamble (81 FR 13040). REGULATION sources are current law; COMMENTARY explains the 2016 rule and can "
        "describe earlier or proposed text. This is general information, not legal advice."
    ),
)

_conn = None


class Passage(BaseModel):
    n: int
    citation_id: str
    citation: str
    kind: str  # REGULATION (current law) or COMMENTARY (2016 preamble)
    url: str
    text: str
    score: float


class Section(BaseModel):
    citation: str
    kind: str
    url: str
    chunks: int
    text: str
    truncated: bool


class Usage(BaseModel):
    model: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


class Answered(BaseModel):
    answer: str | None
    mode: str
    disclaimer: str
    sources: list[Passage]
    usage: Usage


def get_conn():
    """One lazily opened connection per server process; reopened if a failed query left it unusable."""
    global _conn
    if _conn is None or _conn.closed:
        _conn = connect()
    elif _conn.info.transaction_status == TransactionStatus.INERROR:
        _conn.rollback()
    return _conn


def _kind(source: str) -> str:
    return {"eCFR": "REGULATION", "Federal Register": "COMMENTARY"}.get(source, source)


def _citation_id(citation: str) -> str:
    """The part of a citation that names a section: '8 CFR 214.2(f)(10)' from '8 CFR 214.2(f)(10) Practical training'.
    Commentary headings have no shorter form, so their citation is used whole."""
    match = re.match(r"8 CFR 214\.2\(f\)(\(\d+\))?", citation)
    return match.group(0) if match else citation


def _passage(n: int, hit) -> Passage:
    return Passage(
        n=n,
        citation_id=_citation_id(hit.citation),
        citation=hit.citation,
        kind=_kind(hit.source),
        url=hit.source_url,
        text=hit.content,
        score=round(hit.score, 4),
    )


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@mcp.tool()
def search_regulations(query: str, top_k: int = 6) -> list[Passage]:
    """Find the passages of 8 CFR 214.2(f) and the 2016 STEM OPT rule preamble that best match a question or keywords.

    Returns up to top_k passages, best first. Each has the full passage text, its citation, a citation_id you can pass
    to get_section, and kind: REGULATION (current law) or COMMENTARY (the government's explanation, may be dated).
    """
    if not 1 <= top_k <= MAX_TOP_K:
        raise ToolError(f"top_k must be between 1 and {MAX_TOP_K}")
    if len(query.strip()) < 3:
        raise ToolError("query must be at least 3 characters")
    hits = search(get_conn(), query.strip(), top_k, "rerank")
    return [_passage(i, h) for i, h in enumerate(hits, start=1)]


@mcp.tool()
def get_section(citation_id: str) -> Section:
    """Return the full indexed text of one section, e.g. "8 CFR 214.2(f)(10)" (OPT) or the citation of a commentary heading.

    Use the citation_id from search_regulations. A short form like "8 CFR 214.2(f)(10)" matches the section whose
    citation starts with it, such as "8 CFR 214.2(f)(10) Practical training". Text is returned in document order.
    """
    wanted = citation_id.strip()
    if not wanted:
        raise ToolError("citation_id is required")
    escaped = _like_escape(wanted.lower())
    rows = get_conn().execute(
        "SELECT citation, source, source_url, content FROM chunks "
        "WHERE lower(citation) = %s OR lower(citation) LIKE %s ESCAPE '\\' OR lower(citation) LIKE %s ESCAPE '\\' "
        "ORDER BY id",
        (wanted.lower(), escaped + " %", escaped + ",%"),
    ).fetchall()
    if not rows:
        raise ToolError(f"No indexed section matches {wanted!r}. Call search_regulations first and use a citation_id from its results.")
    text = "\n\n".join(r[3] for r in rows)
    truncated = len(text) > MAX_SECTION_CHARS
    return Section(
        citation=rows[0][0],
        kind=_kind(rows[0][1]),
        url=rows[0][2],
        chunks=len(rows),
        text=text[:MAX_SECTION_CHARS],
        truncated=truncated,
    )


@mcp.tool()
def ask(question: str) -> Answered:
    """Answer a question about F-1 / OPT / STEM OPT from the indexed sources, with [n] citations into `sources`.

    Uses the server's configured model. If none is configured it returns the sources with answer null
    (mode "retrieval-only"). If the sources do not contain the answer it says so instead of guessing.
    """
    if len(question.strip()) < 3 or len(question) > 500:
        raise ToolError("question must be between 3 and 500 characters")
    hits = search(get_conn(), question.strip(), 6, "rerank")
    try:
        result = answer(question.strip(), hits)
    except ProviderError as exc:
        raise ToolError(f"The model provider failed: {exc}") from exc  # ToolError is the only error type whose text reaches the client
    return Answered(
        answer=result.text,
        mode=result.mode,
        disclaimer=DISCLAIMER,
        sources=[_passage(i, h) for i, h in enumerate(hits, start=1)],
        usage=Usage(
            model=result.model, input_tokens=result.input_tokens, output_tokens=result.output_tokens, cost_usd=result.cost_usd
        ),
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
