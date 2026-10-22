"""STEM OPT final rule (81 FR 13040, March 11, 2016): the preamble explains the reasoning behind the rule."""
import httpx
from lxml import etree

from app.chunking import pack
from app.sources import Chunk

USER_AGENT = "opt-compass-ingest/0.1 (personal research project)"
DOC_API = "https://www.federalregister.gov/api/v1/documents/2016-04828.json"
CITE = "81 FR 13040 (STEM OPT final rule)"


def fetch_xml() -> tuple[bytes, str]:
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True) as client:
        meta = client.get(DOC_API).json()
        xml = client.get(meta["full_text_xml_url"])
        xml.raise_for_status()
        return xml.content, meta["html_url"]


def build_chunks(xml: bytes, html_url: str) -> list[Chunk]:
    root = etree.fromstring(xml, parser=etree.XMLParser(recover=True))
    heading = "Introduction"
    buckets: list[tuple[str, list[str]]] = [(heading, [])]
    for el in root.iter("HD", "P"):
        text = " ".join("".join(el.itertext()).split())
        if not text:
            continue
        if el.tag == "HD":
            heading = text[:120]
            buckets.append((heading, []))
        else:
            buckets[-1][1].append(text)
    chunks: list[Chunk] = []
    for heading, paras in buckets:
        # The amendatory text under "Authority:" is the 2016 wording of regulation that the eCFR
        # source already carries in its current form; indexing both would serve stale duplicates.
        if heading.startswith("Authority"):
            continue
        for window in pack(paras):
            if len(window) < 200:
                continue
            chunks.append(Chunk("Federal Register", html_url, f"{CITE}, {heading}", f"{CITE}, {heading}\n{window}"))
    return chunks
