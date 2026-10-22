import math
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.config import load_settings


@lru_cache(maxsize=1)
def _model() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=load_settings().rerank_model)


def rerank_scores(question: str, passages: list[str]) -> list[float]:
    """Cross-encoder relevance as a probability in (0, 1), one per passage."""
    return [1.0 / (1.0 + math.exp(-s)) for s in _model().rerank(question, passages)]
