"""Start the MCP server as a subprocess over stdio and call each tool, as Claude Desktop or Claude Code would.

    .venv/bin/python examples/mcp_client_demo.py
The `ask` call needs a model configured in the repo-root .env (LLM_PROVIDER, LLM_MODEL and a key).
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parents[1]


def show(title: str, payload) -> None:
    print(f"\n### {title}")
    print(payload if isinstance(payload, str) else json.dumps(payload, indent=2))


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        cwd=str(ROOT),
        env={**os.environ, "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "gemini"), "LLM_MODEL": os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")},
    )
    async with Client(params) as client:
        show("server", f"{client.server_info.name}")
        tools = (await client.list_tools()).tools
        show("tools/list", [t.name for t in tools])

        found = await client.call_tool("search_regulations", {"query": "does OPT end if I transfer schools", "top_k": 3})
        passages = found.structured_content["result"]
        show("search_regulations(query='does OPT end if I transfer schools', top_k=3)", [
            {k: (v[:110] + "...") if k == "text" else v for k, v in p.items()} for p in passages])

        section = await client.call_tool("get_section", {"citation_id": passages[0]["citation_id"]})
        body = section.structured_content
        show(f"get_section(citation_id={passages[0]['citation_id']!r})", {**body, "text": body["text"][:300] + "..."})

        missing = await client.call_tool("get_section", {"citation_id": "8 CFR 999"})
        show("get_section(citation_id='8 CFR 999')  (is_error=%s)" % missing.is_error, missing.content[0].text)

        asked = await client.call_tool("ask", {"question": "Does OPT authorization end if I transfer to another school?"})
        a = asked.structured_content
        show("ask(question='Does OPT authorization end if I transfer to another school?')", {
            "answer": a["answer"], "mode": a["mode"], "usage": a["usage"],
            "sources": [f"[{s['n']}] {s['kind']}: {s['citation']}" for s in a["sources"]]})


asyncio.run(main())
