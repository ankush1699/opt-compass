"""Pages the user saved by hand (uscis.gov / ice.gov block scripted downloads)."""
from pathlib import Path

from app.chunking import pack
from app.sources import Chunk

MANUAL_DIR = Path(__file__).resolve().parents[3] / "data" / "manual"


def build_chunks(directory: Path = MANUAL_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.txt")):
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or not lines[0].startswith("Source:"):
            continue
        url = lines[0].removeprefix("Source:").strip()
        title = path.stem
        body_start = 1
        if len(lines) > 1 and lines[1].startswith("Title:"):
            title = lines[1].removeprefix("Title:").strip()
            body_start = 2
        for window in pack(lines[body_start:]):
            chunks.append(Chunk("Saved page", url, f"{title} ({url})", f"{title}\n{window}"))
    return chunks
