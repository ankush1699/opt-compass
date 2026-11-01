import json
import queue
import threading
import time
from contextlib import asynccontextmanager

import anyio
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import accesslog
from app.config import load_settings
from app.db import connect
from app.generate import DISCLAIMER, Answer, answer, answer_stream
from app.llm import ProviderError
from app.retrieve import search
from app.security import RateLimiter, parse_keys, require_key
from app.tracing import trace


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=6, ge=1, le=15)
    mode: str = Field(default="rerank", pattern="^(dense|keyword|hybrid|rerank)$")
    stream: bool = Field(default=False, description="true: answer as Server-Sent Events (sources, token..., done)")


class SourceOut(BaseModel):
    n: int
    citation: str
    url: str
    snippet: str
    score: float


class UsageOut(BaseModel):
    model: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None when the price is unknown; 0.0 on the free tier
    generation_s: float
    total_s: float


class AskResponse(BaseModel):
    answer: str | None
    mode: str
    disclaimer: str
    sources: list[SourceOut]
    usage: UsageOut


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    accesslog.configure()
    app.state.api_keys = parse_keys(settings.api_keys)
    app.state.limiter = RateLimiter(settings.rate_limit_per_minute)
    app.state.conn = connect()
    yield
    app.state.conn.close()


app = FastAPI(title="OPT Compass rag-service", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    count = app.state.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    return {"status": "ok", "chunks": count}


def _sources(hits) -> list[SourceOut]:
    return [
        SourceOut(n=i, citation=h.citation, url=h.source_url, snippet=h.content[:400], score=round(h.score, 4))
        for i, h in enumerate(hits, start=1)
    ]


def _usage(result: Answer, started: float) -> UsageOut:
    return UsageOut(
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        generation_s=round(result.latency_s, 3),
        total_s=round(time.perf_counter() - started, 3),
    )


def _log(key: str, req: AskRequest, status: int, started: float, hits=(), result: Answer | None = None, error: str | None = None):
    accesslog.log_request(
        key=key,
        path="/ask",
        status=status,
        stream=req.stream,
        retrieval_mode=req.mode,
        question_chars=len(req.question),
        sources=len(hits),
        answer_mode=result.mode if result else None,
        model=result.model if result else None,
        input_tokens=result.input_tokens if result else 0,
        output_tokens=result.output_tokens if result else 0,
        cost_usd=result.cost_usd if result else None,
        latency_ms=round((time.perf_counter() - started) * 1000),
        error=error,
    )


def _event(name: str, data) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def _stream_worker(req: AskRequest, key: str, started: float, out: queue.Queue, cancelled: threading.Event) -> None:
    """Runs the whole pipeline in ONE thread, so the tracing spans open and close in the thread that made them."""
    hits, result, error = [], None, None
    try:
        with trace("ask", metadata={"retrieval_mode": req.mode, "top_k": str(req.top_k), "stream": "true"}) as root:
            root.update(input={"question": req.question})
            try:
                hits = search(app.state.conn, req.question, req.top_k, req.mode)
            except Exception:  # a failed query must not poison the shared connection
                app.state.conn.rollback()
                error = "retrieval failed"
                root.update(level="ERROR", status_message=error)
                out.put(_event("error", {"detail": error}))
                return
            out.put(_event("sources", [s.model_dump() for s in _sources(hits)]))
            deltas = answer_stream(req.question, hits)
            try:
                for item in deltas:
                    if isinstance(item, str):
                        if cancelled.is_set():
                            error = "client disconnected"
                            return
                        out.put(_event("token", {"text": item}))
                    else:
                        result = item
            finally:
                deltas.close()
            root.update(output={"mode": result.mode, "answer": result.text, "cost_usd": result.cost_usd})
            out.put(_event("done", {"mode": result.mode, "disclaimer": DISCLAIMER, "usage": _usage(result, started).model_dump()}))
    except ProviderError as exc:
        error = str(exc)
        out.put(_event("error", {"detail": error}))
    except Exception as exc:
        error = f"{type(exc).__name__}"
        out.put(_event("error", {"detail": "generation failed"}))
    finally:
        _log(key, req, 200, started, hits, result, error)
        out.put(None)


async def _sse(req: AskRequest, key: str, started: float):
    out: queue.Queue = queue.Queue()
    cancelled = threading.Event()
    threading.Thread(target=_stream_worker, args=(req, key, started, out, cancelled), daemon=True).start()
    try:
        while (item := await anyio.to_thread.run_sync(out.get)) is not None:
            yield item
    finally:
        cancelled.set()


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, key: str = Depends(require_key)):
    started = time.perf_counter()
    if req.stream:
        return StreamingResponse(
            _sse(req, key, started),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    hits, result = [], None
    with trace("ask", metadata={"retrieval_mode": req.mode, "top_k": str(req.top_k)}) as root:
        root.update(input={"question": req.question})
        try:
            hits = search(app.state.conn, req.question, req.top_k, req.mode)
        except Exception as exc:  # a failed query must not poison the shared connection
            app.state.conn.rollback()
            root.update(level="ERROR", status_message="retrieval failed")
            _log(key, req, 500, started, error="retrieval failed")
            raise HTTPException(status_code=500, detail="retrieval failed") from exc
        try:
            result = answer(req.question, hits)
        except ProviderError as exc:
            root.update(level="ERROR", status_message=str(exc))
            _log(key, req, 502, started, hits, error=str(exc))
            raise HTTPException(status_code=502, detail=f"The model provider failed: {exc}") from exc
        root.update(output={"mode": result.mode, "answer": result.text, "cost_usd": result.cost_usd})
    _log(key, req, 200, started, hits, result)
    return AskResponse(
        answer=result.text,
        mode=result.mode,
        disclaimer=DISCLAIMER,
        sources=_sources(hits),
        usage=_usage(result, started),
    )
