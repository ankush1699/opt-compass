from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    source: str
    source_url: str
    citation: str
    content: str
