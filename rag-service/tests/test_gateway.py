import json
import logging

import pytest
from fastapi.testclient import TestClient

from app import main
from app.generate import Answer
from app.llm import ProviderError
from app.retrieve import Hit
from app.security import RateLimiter, authenticate, parse_keys

HIT = Hit(1, "eCFR", "https://example.test/ecfr", "8 CFR 214.2(f)(10) Practical training", "90 days of unemployment " * 30, 0.0321)
QUESTION = {"question": "How many days of unemployment?"}
KEY = {"X-API-Key": "secret-123"}
SECRET_QUESTION = "my very private immigration situation"


class FakeConn:
    def execute(self, *_a, **_k):
        class Cursor:
            def fetchone(self):
                return (7,)

        return Cursor()

    def rollback(self):
        pass

    def close(self):
        pass


def final_answer(**overrides):
    return Answer(**{"text": "Ninety days [1].", "mode": "generated", "model": "test-model", "input_tokens": 1200,
                     "output_tokens": 30, "latency_s": 0.4, "cost_usd": 0.0, "stop_reason": "end_turn", **overrides})


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_KEYS", "tester:secret-123,other:secret-456")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    monkeypatch.setattr(main, "connect", lambda: FakeConn())
    monkeypatch.setattr(main, "search", lambda *_a, **_k: [HIT])
    monkeypatch.setattr(main, "answer", lambda *_a, **_k: final_answer())

    def fake_stream(*_a, **_k):
        yield "Ninety "
        yield "days [1]."
        yield final_answer()

    monkeypatch.setattr(main, "answer_stream", fake_stream)
    with TestClient(main.app) as c:
        yield c


def sse_events(response) -> list[tuple[str, dict]]:
    events, name = [], None
    for line in response.iter_lines():
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            events.append((name, json.loads(line[5:])))
    return events


# ---- auth ----
def test_ask_needs_a_valid_key(client):
    assert client.post("/ask", json=QUESTION).status_code == 401
    assert client.post("/ask", json=QUESTION, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/ask", json=QUESTION, headers=KEY).status_code == 200
    assert client.post("/ask", json=QUESTION, headers={"Authorization": "Bearer secret-456"}).status_code == 200


def test_unauthenticated_response_says_how_to_authenticate(client):
    response = client.post("/ask", json=QUESTION)
    assert response.headers["www-authenticate"] == "Bearer"


def test_health_needs_no_key(client):
    assert client.get("/health").status_code == 200


def test_server_without_keys_refuses_everything(client, monkeypatch):
    monkeypatch.setenv("API_KEYS", "")
    with TestClient(main.app) as unconfigured:
        assert unconfigured.post("/ask", json=QUESTION, headers=KEY).status_code == 503


def test_parse_keys_and_authenticate():
    keys = parse_keys(" a:one , b:two:with-colon,junk,:nokey,noval: ")
    assert keys == {"a": "one", "b": "two:with-colon"}
    assert authenticate(keys, "two:with-colon") == "b"
    assert authenticate(keys, "") is None and authenticate({}, "x") is None


# ---- rate limit ----
def test_rate_limit_is_per_key_and_returns_retry_after(client):
    for _ in range(3):
        assert client.post("/ask", json=QUESTION, headers=KEY).status_code == 200
    blocked = client.post("/ask", json=QUESTION, headers=KEY)
    assert blocked.status_code == 429
    assert 1 <= int(blocked.headers["retry-after"]) <= 60
    assert client.post("/ask", json=QUESTION, headers={"X-API-Key": "secret-456"}).status_code == 200


def test_rejected_keys_do_not_use_up_anyones_allowance(client):
    for _ in range(10):
        client.post("/ask", json=QUESTION, headers={"X-API-Key": "wrong"})
    assert client.post("/ask", json=QUESTION, headers=KEY).status_code == 200


def test_rate_limiter_window_slides():
    now = [0.0]
    limiter = RateLimiter(2, window_s=60, clock=lambda: now[0])
    assert limiter.check("k") is None and limiter.check("k") is None
    assert limiter.check("k") == pytest.approx(60)
    now[0] = 59
    assert limiter.check("k") == pytest.approx(1)
    now[0] = 60
    assert limiter.check("k") is None


# ---- JSON response, cost, logging ----
def test_json_response_reports_usage_and_cost(client):
    body = client.post("/ask", json=QUESTION, headers=KEY).json()
    assert body["answer"] == "Ninety days [1]." and body["sources"][0]["n"] == 1
    usage = body["usage"]
    assert (usage["model"], usage["input_tokens"], usage["output_tokens"], usage["cost_usd"]) == ("test-model", 1200, 30, 0.0)
    assert usage["generation_s"] == 0.4 and usage["total_s"] >= 0


def test_access_log_has_cost_and_key_name_but_no_secret_or_question(client, caplog):
    caplog.set_level(logging.INFO, logger="opt_compass.access")
    client.post("/ask", json={"question": SECRET_QUESTION}, headers=KEY)
    line = next(r.getMessage() for r in caplog.records if r.name == "opt_compass.access")
    entry = json.loads(line)
    assert entry["key"] == "tester" and entry["status"] == 200 and entry["cost_usd"] == 0.0
    assert (entry["input_tokens"], entry["output_tokens"], entry["answer_mode"]) == (1200, 30, "generated")
    assert entry["question_chars"] == len(SECRET_QUESTION)
    assert "secret-123" not in line and SECRET_QUESTION not in line and "Ninety" not in line


def test_provider_failure_is_a_502_and_is_logged(client, monkeypatch, caplog):
    def boom(*_a, **_k):
        raise ProviderError("Gemini returned 503 after 5 attempts")

    monkeypatch.setattr(main, "answer", boom)
    caplog.set_level(logging.INFO, logger="opt_compass.access")
    response = client.post("/ask", json=QUESTION, headers=KEY)
    assert response.status_code == 502 and "503" in response.json()["detail"]
    assert json.loads(caplog.records[-1].getMessage())["status"] == 502


# ---- SSE ----
def test_stream_sends_sources_then_tokens_then_done(client):
    with client.stream("POST", "/ask", json={**QUESTION, "stream": True}, headers=KEY) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = sse_events(response)
    assert [name for name, _ in events] == ["sources", "token", "token", "done"]
    assert events[0][1][0]["citation"].startswith("8 CFR 214.2(f)(10)")
    assert "".join(data["text"] for name, data in events if name == "token") == "Ninety days [1]."
    done = events[-1][1]
    assert done["mode"] == "generated" and "not legal advice" in done["disclaimer"]
    assert done["usage"]["cost_usd"] == 0.0 and done["usage"]["output_tokens"] == 30


def test_stream_is_logged_once_with_cost(client, caplog):
    caplog.set_level(logging.INFO, logger="opt_compass.access")
    with client.stream("POST", "/ask", json={**QUESTION, "stream": True}, headers=KEY) as response:
        sse_events(response)
    entries = [json.loads(r.getMessage()) for r in caplog.records if r.name == "opt_compass.access"]
    assert len(entries) == 1 and entries[0]["stream"] is True and entries[0]["cost_usd"] == 0.0 and entries[0]["error"] is None


def test_stream_needs_a_key_and_counts_against_the_limit(client):
    assert client.post("/ask", json={**QUESTION, "stream": True}).status_code == 401
    for _ in range(3):
        with client.stream("POST", "/ask", json={**QUESTION, "stream": True}, headers=KEY) as response:
            sse_events(response)
    assert client.post("/ask", json={**QUESTION, "stream": True}, headers=KEY).status_code == 429


def test_stream_reports_a_provider_failure_as_an_error_event(client, monkeypatch, caplog):
    def failing(*_a, **_k):
        yield "partial "
        raise ProviderError("Gemini returned 503 after 5 attempts")

    monkeypatch.setattr(main, "answer_stream", failing)
    caplog.set_level(logging.INFO, logger="opt_compass.access")
    with client.stream("POST", "/ask", json={**QUESTION, "stream": True}, headers=KEY) as response:
        events = sse_events(response)
    assert [name for name, _ in events] == ["sources", "token", "error"]
    assert "503" in events[-1][1]["detail"]
    assert "503" in json.loads(caplog.records[-1].getMessage())["error"]


def test_stream_retrieval_failure_is_an_error_event(client, monkeypatch):
    def broken(*_a, **_k):
        raise RuntimeError("db down")

    monkeypatch.setattr(main, "search", broken)
    with client.stream("POST", "/ask", json={**QUESTION, "stream": True}, headers=KEY) as response:
        events = sse_events(response)
    assert events == [("error", {"detail": "retrieval failed"})]
