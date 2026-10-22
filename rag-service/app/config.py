import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    embed_model: str
    llm_model: str
    rerank_model: str
    anthropic_api_key: str
    embed_dim: int
    top_k: int = 6


def load_settings() -> Settings:
    return Settings(
        database_url=os.getenv("DATABASE_URL", "postgresql://compass:compass@localhost:5433/compass"),
        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5"),
        llm_model=os.getenv("LLM_MODEL", "claude-haiku-4-5"),
        rerank_model=os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        embed_dim=int(os.getenv("EMBED_DIM", "768")),
    )
