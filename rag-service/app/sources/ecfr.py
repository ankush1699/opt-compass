"""8 CFR 214.2(f): the regulation that governs F-1 status, OPT and STEM OPT, from the eCFR API."""
import re

import httpx
from lxml import etree

from app.chunking import pack
from app.sources import Chunk

USER_AGENT = "opt-compass-ingest/0.1 (personal research project)"
TITLES_URL = "https://www.ecfr.gov/api/versioner/v1/titles.json"
FULL_URL = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-8.xml?part=214&section=214.2"
SOURCE_URL = "https://www.ecfr.gov/current/title-8/chapter-I/subchapter-B/part-214/section-214.2"

# Regulation rules are short and distinct; small windows keep one rule per chunk so it is not diluted.
ECFR_CHUNK_CHARS = 700
LABEL_RE = re.compile(r"^\(([A-Za-z]{1,4}|\d{1,3})\)\s*")


def _norm(text: str) -> str:
    """Letters and digits only, lowercased, so en/em dashes and spacing never block a heading match."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _is_heading(n: int, text: str, toc: dict[int, str], current: int) -> bool:
    """True when `text` opens top-level paragraph (f)(n).

    The contents table sometimes shortens a heading ("Form I-20 or successor form" vs the body's
    "(2) Student maintenance of Form I-20 or successor form"), so a next-in-sequence paragraph is
    accepted when its opening words contain the table heading. A numbered sub-paragraph such as
    "(11) Site visits" (really (f)(10)(ii)(C)(11)) matches neither rule and stays in (f)(10).
    """
    if n not in toc:
        return False
    if _norm(text).startswith(_norm(f"({n}) {toc[n]}")[:14]):
        return True
    return n == current + 1 and _norm(toc[n]) in _norm(text[:150])


def fetch_section_xml() -> tuple[bytes, str]:
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    with httpx.Client(headers=headers, timeout=60, follow_redirects=True) as client:
        titles = client.get(TITLES_URL).json()["titles"]
        date = next(t for t in titles if t["number"] == 8)["latest_issue_date"]
        resp = client.get(FULL_URL.format(date=date))
        resp.raise_for_status()
        return resp.content, date


def _paragraph_contents(root) -> dict[int, str]:
    """Map N -> heading from the 'Paragraph Contents' table that lists (f)(1)...(f)(N)."""
    for table in root.xpath("//TABLE[CAPTION]"):
        caption = " ".join("".join(table.xpath("CAPTION")[0].itertext()).split())
        if "Paragraph (f)" in caption and "Paragraph Contents" in caption:
            toc: dict[int, str] = {}
            for cell in table.xpath(".//TD"):
                m = re.match(r"^\((\d+)\)\s+(.+)", " ".join("".join(cell.itertext()).split()))
                if m:
                    toc[int(m.group(1))] = m.group(2)
            return toc
    raise ValueError("eCFR 214.2(f) paragraph contents table not found")


def parse_f_paragraphs(xml: bytes) -> list[tuple[int, str]]:
    """Return (top_level_number, text) for each paragraph of 214.2(f).

    Only the top-level (f)(N) is tracked, pinned by the section's own contents table. Deeper outline
    levels ((ii), (C), (1)...) are not reliably recoverable from the flat XML, so they are left in the
    paragraph text exactly as published instead of being guessed into the citation.
    """
    root = etree.fromstring(xml)
    toc = _paragraph_contents(root)
    paras = [
        " ".join("".join(p.itertext()).split())
        for p in root.xpath("//P[not(ancestor::TABLE)]")
    ]
    start = next(i for i, t in enumerate(paras) if t.startswith("(f) Students in colleges"))
    end = next(i for i, t in enumerate(paras) if i > start and re.match(r"^\(g\) ", t))
    current = 0
    out: list[tuple[int, str]] = []
    for text in paras[start + 1 : end]:
        match = LABEL_RE.match(text)
        if match and match.group(1).isdigit():
            n = int(match.group(1))
            if _is_heading(n, text, toc, current):
                current = n
        out.append((current, text))
    return out


def build_chunks(xml: bytes, as_of: str) -> list[Chunk]:
    root = etree.fromstring(xml)
    toc = _paragraph_contents(root)
    grouped: dict[int, list[str]] = {}
    for n, text in parse_f_paragraphs(xml):
        grouped.setdefault(n, []).append(text)
    chunks: list[Chunk] = []
    for n, texts in grouped.items():
        citation = f"8 CFR 214.2(f)({n}) {toc[n]}" if n else "8 CFR 214.2(f)"
        for window in pack(texts, max_chars=ECFR_CHUNK_CHARS):
            content = f"{citation} (eCFR as of {as_of})\n{window}"
            chunks.append(Chunk("eCFR", SOURCE_URL, citation, content))
    return chunks
