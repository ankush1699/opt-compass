"""Langfuse tracing that disappears when it is not configured.

Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST (the self-hosted instance from
observability/) to trace every request: retrieval, rerank and generation, with tokens, cost and
latency. With no keys the helpers do nothing, so tests and local runs need no Langfuse.
"""
import os
from contextlib import contextmanager
from typing import Any, Iterator


class _Null:
    def update(self, **_kwargs: Any) -> None:
        pass


def enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


@contextmanager
def observe(name: str, as_type: str = "span", **fields: Any) -> Iterator[Any]:
    """Open a Langfuse observation as a child of the current one; yields an object with .update(**fields)."""
    if not enabled():
        yield _Null()
        return
    from langfuse import get_client

    with get_client().start_as_current_observation(name=name, as_type=as_type, **fields) as obs:
        yield obs


@contextmanager
def trace(name: str, **attributes: Any) -> Iterator[Any]:
    """Root observation for one request; attributes (metadata, tags) propagate to every child."""
    if not enabled():
        yield _Null()
        return
    from langfuse import get_client, propagate_attributes

    with get_client().start_as_current_observation(name=name, as_type="span") as root:
        with propagate_attributes(trace_name=name, **attributes):
            yield root


def flush() -> None:
    if enabled():
        from langfuse import get_client

        get_client().flush()
