import psycopg
from pgvector.psycopg import register_vector

from app.config import load_settings

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL,
    source_url  TEXT NOT NULL,
    citation    TEXT NOT NULL,
    content     TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding   vector({dim}) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (source_url, content_hash)
);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def connect() -> psycopg.Connection:
    settings = load_settings()
    # The extension must exist before the vector type can be registered on the connection.
    with psycopg.connect(settings.database_url, autocommit=True) as boot:
        boot.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA.format(dim=load_settings().embed_dim))
    conn.commit()
