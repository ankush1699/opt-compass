from functools import lru_cache

from fastembed import TextEmbedding

from app.config import load_settings

# bge models retrieve better when the query (not the passages) carries this instruction.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    return TextEmbedding(model_name=load_settings().embed_model)


def embed_passages(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _model().embed(texts)]


def embed_query(text: str) -> list[float]:
    return next(iter(_model().embed([QUERY_PREFIX + text]))).tolist()
