from fastapi.testclient import TestClient

from app import main
from app.generate import Answer
from app.retrieve import Hit, _or_query


def test_or_query_drops_stopwords_and_dedupes():
    assert _or_query("How many days of unemployment do I get, unemployment?") == "many | days | unemployment | get"


def test_or_query_is_empty_when_only_stopwords():
    assert _or_query("what is the") == ""


class FakeConn:
    def execute(self, *_args, **_kwargs):
        class Cursor:
            def fetchone(self):
                return (7,)

        return Cursor()

    def rollback(self):
        pass

    def close(self):
        pass


def _client(monkeypatch, hits, generated=None):
    monkeypatch.setattr(main, "connect", lambda: FakeConn())
    monkeypatch.setattr(main, "search", lambda *_a, **_k: hits)
    monkeypatch.setattr(main, "answer", lambda *_a, **_k: generated or Answer(None, "retrieval-only"))
    monkeypatch.setenv("API_KEYS", "tester:secret-123")
    return TestClient(main.app, headers={"X-API-Key": "secret-123"})


HIT = Hit(1, "eCFR", "https://example.test/ecfr", "8 CFR 214.2(f)(10) Practical training", "text " * 200, 0.0321)


def test_health_reports_chunk_count(monkeypatch):
    with _client(monkeypatch, []) as client:
        assert client.get("/health").json() == {"status": "ok", "chunks": 7}


def test_ask_returns_cited_sources_and_disclaimer(monkeypatch):
    with _client(monkeypatch, [HIT], Answer("90 days [1]", "generated")) as client:
        body = client.post("/ask", json={"question": "How many days of unemployment?"}).json()
    assert body["answer"] == "90 days [1]"
    assert body["mode"] == "generated"
    assert "not legal advice" in body["disclaimer"]
    assert body["sources"][0]["n"] == 1
    assert body["sources"][0]["citation"].startswith("8 CFR 214.2(f)(10)")
    assert len(body["sources"][0]["snippet"]) <= 400


def test_ask_without_llm_key_is_retrieval_only(monkeypatch):
    with _client(monkeypatch, [HIT]) as client:
        body = client.post("/ask", json={"question": "How many days of unemployment?"}).json()
    assert body["answer"] is None and body["mode"] == "retrieval-only"
    assert len(body["sources"]) == 1


def test_ask_rejects_bad_input(monkeypatch):
    with _client(monkeypatch, []) as client:
        assert client.post("/ask", json={"question": "hi"}).status_code == 422
        assert client.post("/ask", json={"question": "valid question", "mode": "nope"}).status_code == 422
