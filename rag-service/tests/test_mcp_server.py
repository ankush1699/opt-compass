import asyncio

import pytest
from mcp import Client

from app import mcp_server
from app.llm import ProviderError
from app.generate import Answer
from app.retrieve import Hit

REG = Hit(1, "eCFR", "https://example.test/ecfr", "8 CFR 214.2(f)(10) Practical training", "90 days of unemployment", 0.91)
COM = Hit(2, "Federal Register", "https://example.test/fr", "81 FR 13040 (STEM OPT final rule), iii. Unemployment Limits", "150 days", 0.5)


class FakeConn:
    def __init__(self, rows=()):
        self.rows, self.sql = list(rows), []

    def execute(self, sql, params=()):
        self.sql.append((sql, params))
        rows = self.rows

        class Cursor:
            def fetchall(self):
                return rows

        return Cursor()


@pytest.fixture
def server(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(mcp_server, "get_conn", lambda: conn)
    monkeypatch.setattr(mcp_server, "search", lambda *_a, **_k: [REG, COM])
    monkeypatch.setattr(
        mcp_server, "answer",
        lambda *_a, **_k: Answer("90 days [1].", "generated", "test-model", 1000, 20, 0.5, 0.0, "end_turn"),
    )
    mcp_server.fake_conn = conn
    return mcp_server.mcp


def call(server, tool, arguments):
    async def go():
        async with Client(server) as client:
            return await client.call_tool(tool, arguments)

    return asyncio.run(go())


def test_server_lists_the_three_tools_with_schemas(server):
    async def go():
        async with Client(server) as client:
            return (await client.list_tools()).tools

    tools = {t.name: t for t in asyncio.run(go())}
    assert set(tools) == {"search_regulations", "get_section", "ask"}
    assert tools["search_regulations"].input_schema["required"] == ["query"]
    assert set(tools["search_regulations"].input_schema["properties"]) == {"query", "top_k"}
    assert tools["get_section"].input_schema["required"] == ["citation_id"]
    assert tools["ask"].input_schema["required"] == ["question"]
    assert all(t.description for t in tools.values())


def test_search_regulations_returns_ranked_passages_with_kind_and_citation_id(server):
    result = call(server, "search_regulations", {"query": "unemployment days", "top_k": 2})
    assert not result.is_error
    first, second = result.structured_content["result"]
    assert (first["n"], first["kind"], first["citation_id"]) == (1, "REGULATION", "8 CFR 214.2(f)(10)")
    assert first["text"] == "90 days of unemployment" and first["url"] == "https://example.test/ecfr"
    assert (second["kind"], second["citation_id"]) == ("COMMENTARY", "81 FR 13040 (STEM OPT final rule), iii. Unemployment Limits")


@pytest.mark.parametrize("arguments", [{"query": "ok question", "top_k": 0}, {"query": "ok question", "top_k": 99}, {"query": " "}])
def test_search_regulations_rejects_bad_arguments(server, arguments):
    assert call(server, "search_regulations", arguments).is_error


def test_get_section_joins_chunks_in_order_and_matches_the_short_citation(server):
    mcp_server.fake_conn.rows = [
        ("8 CFR 214.2(f)(10) Practical training", "eCFR", "https://example.test/ecfr", "chunk one"),
        ("8 CFR 214.2(f)(10) Practical training", "eCFR", "https://example.test/ecfr", "chunk two"),
    ]
    result = call(server, "get_section", {"citation_id": "8 CFR 214.2(f)(10)"})
    body = result.structured_content
    assert body["text"] == "chunk one\n\nchunk two" and body["chunks"] == 2 and body["kind"] == "REGULATION"
    assert body["citation"] == "8 CFR 214.2(f)(10) Practical training" and body["truncated"] is False
    sql, params = mcp_server.fake_conn.sql[0]
    assert params == ("8 CFR 214.2(f)(10)".lower(), "8 cfr 214.2(f)(10) %", "8 cfr 214.2(f)(10),%")


def test_get_section_escapes_like_wildcards_and_reports_unknown_ids(server):
    result = call(server, "get_section", {"citation_id": "100%_off"})
    assert result.is_error
    assert mcp_server.fake_conn.sql[0][1][1] == "100\\%\\_off %"
    assert "search_regulations" in result.content[0].text


def test_get_section_truncates_very_long_sections(server):
    mcp_server.fake_conn.rows = [("8 CFR 214.2(f)(10) Practical training", "eCFR", "u", "x" * (mcp_server.MAX_SECTION_CHARS + 50))]
    body = call(server, "get_section", {"citation_id": "8 CFR 214.2(f)(10)"}).structured_content
    assert body["truncated"] is True and len(body["text"]) == mcp_server.MAX_SECTION_CHARS


def test_ask_returns_the_answer_sources_disclaimer_and_usage(server):
    body = call(server, "ask", {"question": "How many days of unemployment?"}).structured_content
    assert body["answer"] == "90 days [1]." and body["mode"] == "generated"
    assert [s["n"] for s in body["sources"]] == [1, 2]
    assert "not legal advice" in body["disclaimer"]
    assert body["usage"] == {"model": "test-model", "input_tokens": 1000, "output_tokens": 20, "cost_usd": 0.0}


def test_ask_without_a_model_returns_sources_and_no_answer(server, monkeypatch):
    monkeypatch.setattr(mcp_server, "answer", lambda *_a, **_k: Answer(None, "retrieval-only"))
    body = call(server, "ask", {"question": "How many days of unemployment?"}).structured_content
    assert body["answer"] is None and body["mode"] == "retrieval-only" and len(body["sources"]) == 2


def test_ask_reports_a_provider_failure_to_the_client(server, monkeypatch):
    def fail(*_a, **_k):
        raise ProviderError("No model is configured: set LLM_MODEL")

    monkeypatch.setattr(mcp_server, "answer", fail)
    result = call(server, "ask", {"question": "How many days of unemployment?"})
    assert result.is_error and "No model is configured" in result.content[0].text


def test_ask_rejects_a_too_short_question(server):
    assert call(server, "ask", {"question": "hi"}).is_error


def test_citation_id_shortens_regulation_citations_only():
    assert mcp_server._citation_id("8 CFR 214.2(f)(1) Admission of student") == "8 CFR 214.2(f)(1)"
    assert mcp_server._citation_id("8 CFR 214.2(f)") == "8 CFR 214.2(f)"
    assert mcp_server._citation_id("81 FR 13040 (STEM OPT final rule), 1. OPT Background") == "81 FR 13040 (STEM OPT final rule), 1. OPT Background"
