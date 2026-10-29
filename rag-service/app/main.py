from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.db import connect
from app.generate import DISCLAIMER, answer
from app.retrieve import search
from app.tracing import trace


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=6, ge=1, le=15)
    mode: str = Field(default="rerank", pattern="^(dense|keyword|hybrid|rerank)$")


class SourceOut(BaseModel):
    n: int
    citation: str
    url: str
    snippet: str
    score: float


class AskResponse(BaseModel):
    answer: str | None
    mode: str
    disclaimer: str
    sources: list[SourceOut]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.conn = connect()
    yield
    app.state.conn.close()


app = FastAPI(title="OPT Compass rag-service", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    count = app.state.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    return {"status": "ok", "chunks": count}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    with trace("ask", metadata={"retrieval_mode": req.mode, "top_k": str(req.top_k)}) as root:
        root.update(input={"question": req.question})
        try:
            hits = search(app.state.conn, req.question, req.top_k, req.mode)
        except Exception as exc:  # a failed query must not poison the shared connection
            app.state.conn.rollback()
            root.update(level="ERROR", status_message="retrieval failed")
            raise HTTPException(status_code=500, detail="retrieval failed") from exc
        result = answer(req.question, hits)
        root.update(output={"mode": result.mode, "answer": result.text, "cost_usd": result.cost_usd})
    return AskResponse(
        answer=result.text,
        mode=result.mode,
        disclaimer=DISCLAIMER,
        sources=[
            SourceOut(n=i, citation=h.citation, url=h.source_url, snippet=h.content[:400], score=round(h.score, 4))
            for i, h in enumerate(hits, start=1)
        ],
    )
