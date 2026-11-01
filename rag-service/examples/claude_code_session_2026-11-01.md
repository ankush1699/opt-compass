# opt-compass MCP server inside Claude Code, 2026-11-01

Claude Code 2.1.316 on macOS, server registered with `claude mcp add` (README, "MCP server"), `claude mcp list` showed it connected.
This file is what the user pasted from the terminal. It is Claude Code's replies, not the raw tool output, and the first reply
below was cut by terminal line-wrapping when it was copied, so it is summarised rather than quoted.

## 1. search_regulations

Prompt: "Use the opt-compass search_regulations tool to find what the regulation says about the 90-day unemployment limit on OPT. Quote the citation."

Result (summary): the tool was called once and its top passage was the paragraph on periods of unemployment during post-completion
OPT (90 days; 150 days across the whole OPT period for a STEM extension), cited as `8 CFR 214.2(f)(10) Practical training`,
labelled REGULATION. Claude Code noted that the citation stops at the section even though the paragraph is (E) under it.

## 2. ask, first attempt: failed

The tool returned "Error executing tool ask" with no detail. Cause, found by Claude Code and confirmed here: `LLM_MODEL=` was empty in
`.env`, so the request to Gemini carried no model name (404), and the server hid the provider error. Fixed in commit 4fa8ae8 (empty
model name now raises a clear error; provider failures reach the client as a readable tool error).

## 3. ask, after setting LLM_MODEL=gemini-3.5-flash-lite and restarting Claude Code

Prompt: "Use the opt-compass ask tool: if I transfer schools, does my OPT end?"

Claude Code's reply, verbatim:

> Yes. Transferring schools ends your OPT. Under 8 CFR 214.2(f)(10), "authorization to engage in optional practical training employment is automatically terminated when the student transfers to another school or begins study at another educational level." That's the sentence the tool's answer rests on.
>
> The tool also returned a related rule, 8 CFR 214.2(f)(8). If you're on post-completion or STEM OPT, you must be able to begin or resume classes at the new school within 5 months. The 5 months run from the date you transfer out, or from the date your OPT authorization ends, whichever comes first.
>
> The tool's answer was one sentence. It cited the transfer-termination rule, but the (f)(8) detail above comes from the other source excerpts it returned. It doesn't explain what happens to a pending OPT application or a STEM extension. Check those with your DSO before you move.

Not exercised in this session: `get_section`.
