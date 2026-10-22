import hashlib


def pack(paragraphs: list[str], max_chars: int = 1800) -> list[str]:
    """Greedily join paragraphs into windows of at most max_chars (a long paragraph stays whole)."""
    windows: list[str] = []
    current: list[str] = []
    size = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current and size + len(para) > max_chars:
            windows.append("\n".join(current))
            current, size = [], 0
        current.append(para)
        size += len(para) + 1
    if current:
        windows.append("\n".join(current))
    return windows


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
