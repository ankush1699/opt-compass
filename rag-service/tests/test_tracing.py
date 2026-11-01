from app import tracing


def test_tracing_is_a_noop_without_langfuse_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert not tracing.enabled()
    with tracing.trace("ask", metadata={"k": "v"}) as root:
        root.update(output="ignored")
        with tracing.observe("generate", as_type="generation", model="m") as span:
            span.update(usage_details={"input": 1})
    tracing.flush()  # must not raise or try to connect


def test_tracing_needs_both_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert not tracing.enabled()
