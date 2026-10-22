"""Fetch the sources, chunk, embed and upsert into Postgres. Safe to re-run (content-hash dedupe)."""
import sys
from pathlib import Path

from app.chunking import content_hash
from app.db import connect, init_schema
from app.embeddings import embed_passages
from app.sources import Chunk, ecfr, federal_register, manual

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def collect() -> list[Chunk]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    xml, as_of = ecfr.fetch_section_xml()
    (RAW_DIR / "ecfr_214_2.xml").write_bytes(xml)
    chunks = ecfr.build_chunks(xml, as_of)
    print(f"eCFR 8 CFR 214.2(f) as of {as_of}: {len(chunks)} chunks")

    fr_xml, fr_url = federal_register.fetch_xml()
    (RAW_DIR / "fr_2016_04828.xml").write_bytes(fr_xml)
    fr_chunks = federal_register.build_chunks(fr_xml, fr_url)
    print(f"Federal Register STEM OPT rule: {len(fr_chunks)} chunks")

    saved = manual.build_chunks()
    print(f"Manually saved pages: {len(saved)} chunks")
    return chunks + fr_chunks + saved


def main() -> None:
    chunks = collect()
    conn = connect()
    init_schema(conn)
    known = {row[0] for row in conn.execute("SELECT content_hash FROM chunks")}
    fresh = [c for c in chunks if content_hash(c.content) not in known]
    print(f"{len(fresh)} new chunks to embed ({len(chunks) - len(fresh)} already stored)")
    batch = 64
    for i in range(0, len(fresh), batch):
        part = fresh[i : i + batch]
        vectors = embed_passages([c.content for c in part])
        with conn.cursor() as cur:
            for chunk, vec in zip(part, vectors):
                cur.execute(
                    "INSERT INTO chunks (source, source_url, citation, content, content_hash, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (source_url, content_hash) DO NOTHING",
                    (chunk.source, chunk.source_url, chunk.citation, chunk.content, content_hash(chunk.content), vec),
                )
        conn.commit()
        print(f"  embedded {min(i + batch, len(fresh))}/{len(fresh)}")
    total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print(f"done: {total} chunks in database")


if __name__ == "__main__":
    sys.exit(main())
