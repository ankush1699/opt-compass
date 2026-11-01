# OPT Compass

Cited answers to F-1 / OPT / STEM OPT questions, grounded in the primary sources rather than forum posts:
the current text of **8 CFR 214.2(f)** (eCFR API) and the **STEM OPT final rule, 81 FR 13040** (Federal Register API).

> General information from public regulations, not legal advice. Confirm anything important with your school's DSO or an immigration attorney.

**Status: milestone 1.5 (answer-quality evaluation and tracing) plus a round of retrieval and prompt fixes.** Only `rag-service/` and `observability/` exist. Milestone 2 (gateway, MCP server, agent) is built; its evaluation is partial (below). CI and AWS deployment are milestone 3 and are not built.

## Feature status

| feature | status |
|---|---|
| Ingestion from eCFR and Federal Register APIs | built and tested |
| Retrieval (dense, keyword, hybrid, rerank) | built and tested; retrieval metrics below |
| Abbreviation expansion (OPT, DSO, SEVIS...) in retrieval | built and tested; helps the dev set, mixed on 11 held-out questions, see below |
| Sources labelled REGULATION or COMMENTARY in the prompt | built and tested; only partly re-evaluated after the change (see "After the fixes") |
| `POST /ask` with cited answers | built and tested |
| Gateway: API-key auth, per-key rate limit, SSE streaming, cost per request, access log | built and tested (unit tests); live-checked once each on 2026-11-01: SSE through Gemini, 401, 429 with `Retry-After`; the Anthropic streaming path has never been run against the real API |
| Answer-quality eval (faithfulness, citation accuracy, abstention) | built and tested; one live run, results below |
| Judge validation by hand-labelling | built and tested; 18 claims labelled, results below |
| Judge sensitivity probe (`eval/judge_probe.py`) | built and tested; one live run on 11 cases, results below |
| Langfuse tracing (self-hosted) | built and tested for retrieval and rerank spans; generation token and cost recording verified only against a fake model reply, not a live traced run |
| MCP server (`search_regulations`, `get_section`, `ask`) | built and tested; run end to end over stdio with the official MCP client (transcript below); not yet tried inside Claude Desktop or Claude Code |
| LangGraph agent with step limit and refusal path | built and tested; one live run on 10 multi-part questions and 8 refusal probes (below); faithfulness of its answers not yet judged |
| Dockerfiles, CI, AWS deployment | planned (M3) |

## What works today

- **Ingestion** (`python -m app.ingest`): fetches both sources, chunks them, embeds them and upserts into PostgreSQL + pgvector. Re-running is idempotent (content-hash dedupe).
- **Retrieval**: dense (pgvector cosine), keyword (Postgres full text), hybrid (reciprocal rank fusion) and `rerank` (hybrid pool re-scored by a cross-encoder). Regulation text is weighted above Federal Register commentary.
- **API** (FastAPI): `POST /ask` returns the sources with citations and snippets. With a model configured (`LLM_PROVIDER`, `LLM_MODEL`, and the matching key) it also returns a grounded answer whose sentences cite `[n]`, or the sentence "The indexed sources do not answer this."; without one it runs in retrieval-only mode.
- **Retrieval evaluation** (`python -m eval.run_eval`): 28 questions, each with the exact phrase the answer-bearing passage must contain.
- **Answer evaluation** (`python -m eval.answer_eval`): 28 answerable plus 8 unanswerable questions, graded by an LLM judge claim by claim.

### Retrieval results (`BAAI/bge-base-en-v1.5`)

Dev set: 28 questions, the set the chunk size and source weights were tuned on. Held-out set: 11 questions (`eval/questions_heldout.jsonl`) written afterwards and not used for any tuning. Commands: `python -m eval.run_eval` and `python -m eval.run_eval --heldout`, run 2026-11-01.

| mode (dev set) | any-source hit@1 | hit@6 | primary-source hit@1 | hit@6 |
|---|---|---|---|---|
| dense | 0.43 | 0.89 | 0.43 | 0.86 |
| keyword | 0.04 | 0.50 | 0.04 | 0.43 |
| hybrid | 0.39 | 0.86 | 0.39 | 0.82 |
| **rerank (default)** | **0.82** | **0.96** | **0.79** | **0.93** |

*any-source*: a passage from the regulation or the commentary counts. *primary-source*: questions the regulation answers must be answered by an eCFR chunk.

| rerank (default) | questions | any hit@1 | any hit@6 | primary hit@1 | primary hit@6 |
|---|---|---|---|---|---|
| dev set, before the abbreviation fix | 28 | 0.68 | 0.93 | 0.64 | 0.79 |
| dev set, after | 28 | 0.82 | 0.96 | 0.79 | 0.93 |
| held-out, before | 11 | 0.73 | 1.00 | 0.73 | 1.00 |
| held-out, after | 11 | 0.64 | 0.91 | 0.64 | 0.91 |
| both sets, before | 39 | 27 of 39 | 37 of 39 | 26 of 39 | 33 of 39 |
| both sets, after | 39 | 30 of 39 | 37 of 39 | 29 of 39 | 36 of 39 |

The "before" rows are the same code with abbreviation expansion switched off (the dev numbers match the original README). The expansion helps on the dev set and hurts on the held-out set (one question lost at hit@6, one at hit@1), and it made the non-default modes clearly worse (dense hit@1 0.64 to 0.43 on the dev set; before-numbers for those modes on the held-out set were not measured). It is kept because it fixes a diagnosed failure, but the dev improvement is not independent evidence: the fix was found by looking at a dev question, and 11 held-out questions are too few to tell a regression from noise.

What the experiments showed (each change was measured on the same set):

| change | effect |
|---|---|
| 700-char regulation chunks instead of 1,800 | dense hit@6 0.57 → 0.64 (one rule per chunk) |
| down-weight commentary vs. regulation | dense hit@6 0.64 → 0.71; the 425 preamble chunks otherwise crowd out the 125 regulation chunks |
| `bge-small` (384d) → `bge-base` (768d) | any-source dense hit@6 0.86 → 0.93 |
| hybrid fusion | *worse* than dense alone; the keyword leg adds noise on a corpus this small, so it is kept only as an option |
| spell out abbreviations (OPT, DSO, SEVIS...) in the query used to find candidates | q008's answer chunk went from rank 146 of 550 to the top 6; the cross-encoder still scores the user's own words (expanding there too measured worse: dev hit@1 0.75 vs 0.82, held-out 0.55 vs 0.64) |

**Caveats, stated plainly.** 28 questions is a development set, and the chunk size, source weights and the abbreviation list were chosen while looking at it, so treat the dev numbers as a baseline and not a generalisation claim. The held-out set is small (11) and mostly questions without abbreviations, so it mainly checks that the change does not break unrelated questions.

## Answer quality (one live run)

Command, run between 2026-10-31 and 2026-11-01 (it was resumed once after the free tier's daily request limit was reached, so the 36 rows come from one configuration across two sessions):

```bash
cd rag-service
LLM_MODEL=gemini-3.5-flash-lite JUDGE_MODEL=gemini-3.6-flash \
  .venv/bin/python -m eval.answer_eval run --out eval/results/gemini-3.5-flash-lite__judge-gemini-3.6-flash.jsonl
.venv/bin/python -m eval.answer_eval report eval/results/gemini-3.5-flash-lite__judge-gemini-3.6-flash.jsonl
```

Generator `gemini-3.5-flash-lite`, judge `gemini-3.6-flash` (a larger model, but the same vendor), both on Google's free tier, retrieval mode `rerank`, top 6 passages, temperature 0. Raw per-claim verdicts are in `rag-service/eval/results/`.

| metric | value | what it means |
|---|---|---|
| faithfulness | 1.00 (45 of 45 claims) | every claim is supported by some retrieved passage, per the judge |
| citation precision | 1.00 | every cited `[n]` passage supports its claim, per the judge |
| citation coverage | 0.93 (42 of 45) | share of claims that carry a citation |
| invalid citations | 0.00 | `[n]` always pointed at a passage that was shown |
| abstained on unanswerable | 8 of 8 | the model said "The indexed sources do not answer this." |
| false abstentions | 2 of 28 (0.07) | answerable questions it refused |
| generation latency | median 2.2 s, p95 5.5 s | free tier, includes API round trip |
| cost | 0.00 USD | free tier; not a list-price estimate |

The quality gate (`faithfulness >= 0.90`, `citation precision >= 0.85`, `abstain rate on unanswerable >= 0.80`, `false abstain rate <= 0.15`) **passes**. The thresholds were set before the run, but they were not calibrated against any other system, so read the pass as "no obvious failure on these 36 questions" and not as a quality guarantee.

### After the fixes (partial re-evaluation, 2026-11-01)

The 36-row table above is the configuration *before* the fixes. Re-judging all 36 would take two days of the free tier's 20-requests-per-day judge limit, so only part was re-done:

| check | command | result |
|---|---|---|
| all 36 questions re-generated, no judge | `LLM_MODEL=gemini-3.5-flash-lite python -m eval.answer_eval run --no-judge --out eval/results/fixes__generate-only.jsonl` | abstained on unanswerable 8 of 8 (unchanged); false abstentions 1 of 28 (was 2 of 28): q008 now answers, q020 still abstains |
| the 3 answers that changed, judged | `... run --only q008,q025,q026 --out eval/results/fixes__judged-q008-q025-q026.jsonl` | q008: 1 claim, supported and cited. q026: 1 claim, supported, but carries no citation. q025: 6 claims, 5 supported and cited; the lead-in "The government's reasoning ... include:" has no citation and the judge called it unsupported. The answer now states only "April 1" |

What this does and does not show. The re-generation is one run of a model that is not fully deterministic even at temperature 0: when I changed the wording of the source labels, q026 flipped from answered to abstained and back, and q025 mentioned the old "October 1" date (attributed to the commentary) in three of four runs with the final wording and not in the fourth, which is the one that was judged. So "q025 fixed" means "improved and sometimes clean", not "solved". The other 33 answers were not judged again, so faithfulness 1.00 (45 of 45) still describes the pre-fix run only.

### How far to trust the judge

| check | result |
|---|---|
| 18 claims from 12 answers labelled by hand | `supported_by_any`: raw agreement 0.94, Cohen's kappa 0.0; `supported_by_cited`: raw agreement 0.94, kappa 0.77 |
| judge sensitivity probe, 11 cases (run 2026-11-01) | caught 6 of 6 corrupted claims, 0 of 4 untouched controls wrongly flagged, caught 1 of 1 wrong citation |

Read this table carefully. The judge marked all 45 claims as supported, so on that axis there is nothing for kappa to measure: kappa 0.0 does not mean "random", it means the judge never varied, and 0.94 agreement is simply 17 of 18 labels matching. The one disagreement (q025, claim 2) is a case the judge missed, and it is the only evidence in this repo that the judge can be too lenient. The labeller was Claude, not a human, and had seen the judge's overall result before labelling, so the labels are neither blind nor independent of model family. The labels are in `rag-service/eval/manual/labels.jsonl` and anyone can re-check them with `python -m eval.answer_eval sample <run>` and `agreement <run> <labels>`.

`eval/judge_probe.py` corrupts real claims (wrong number, flipped answer, invented source, wrong citation) and checks whether the real judge flags them, with untouched claims as controls. Its logic is unit-tested against a lenient and a strict fake judge. Live run, 2026-11-01, judge `gemini-3.6-flash`, against the 36-row run above:

```bash
cd rag-service
JUDGE_MODEL=gemini-3.6-flash .venv/bin/python -m eval.judge_probe \
  eval/results/gemini-3.5-flash-lite__judge-gemini-3.6-flash.jsonl --out eval/results/judge_probe.json
```

It flagged all 6 corrupted claims (60 days for 90, 30 days for 90, "no recommendation needed", twelve months for six, "replace workers after 120 days", an invented USCIS URL), flagged none of the 4 controls, and caught the 1 claim given a wrong citation. Per-case verdicts and the judge's reasons are in `rag-service/eval/results/judge_probe.json`.

This is a floor on what the judge can do, not a rate: 11 cases, written by hand, and each corruption is blatant (a changed number, a flipped yes/no, an invented source). It says nothing about subtle errors like the q025 case above, which the judge did accept. It also shows the judge can say "unsupported", so the 45 of 45 result is not an artefact of a judge that always says yes.

## Known failures

- **q008 (transfer ends OPT?): fixed by abbreviation expansion, with a caveat.** The chunk saying authorization "is automatically terminated when the student transfers" spells out "optional practical training" and never says "OPT", and it ranked 146th of 550 for the question. Spelling out OPT when searching puts it in the top 6 and the model now answers. The same change made other retrieval modes and one held-out question worse (see the retrieval results).
- **q020 (travel on post-completion OPT?): still a generation failure.** Passage 2 contains the answer and the model still says the sources do not answer, in every run so far (five). This is the small generator being too cautious, and a prompt change to push it harder risks the 8 of 8 abstention result, so it is left alone.
- **Uncited claims (q014, q023, q027).** Three claims carry no `[n]`, and the judge correctly reported them as supported but not by a citation. One of them is the lead-in sentence of a list.
- **Old commentary against current regulation (q025): mitigated, not solved.** Before the fix the answer said the cap-gap extension lasts "until April 1 ... or until the beginning of the new fiscal year (October 1)": the current regulation says April 1, the 2016 preamble describes the older October 1 rule, and the judge accepted the mix. The prompt now labels each source REGULATION (current law) or COMMENTARY and tells the model to follow the regulation on conflicts. Since then the answer attributes April 1 to the regulation and, in three of four runs, also mentions October 1 as the commentary's description; one run states only April 1. The corpus still contains both and nothing in the data marks the commentary as superseded.
- **Prompt wording is fragile.** A first version of the source labels ("may describe earlier or proposed text") made the model abstain on q026, a "why" question that only the commentary answers. Softening the wording fixed it. With a small free-tier model, treat any prompt change as needing a re-run.
- **The generator writes very short answers** (mean about 54 output tokens). Short answers contain fewer claims, which makes a faithfulness score of 1.00 easier to reach. It is not evidence that a more verbose model would score the same.
- **Judge and generator are both Google models**, so shared blind spots and self-preference are possible, and the judge never marked any of the 45 claims unsupported.
- **Single run, no variance estimate.** One configuration, one pass, 36 questions. There are no confidence intervals and the retrieval and answer questions overlap, so these are development-set numbers.
- **Free-tier data use.** Google's pricing page lists free-tier content as used to improve its products. Only public regulation text and synthetic questions were sent.
- **Free-tier operations.** The first attempt failed on `503 high demand` for the first two models chosen, and the judge allows only 20 requests per day, which is why the run spans two sessions.

## Gateway (`POST /ask`)

Set `API_KEYS=name:secret,name2:secret2` (repo-root `.env` or the environment) and optionally `RATE_LIMIT_PER_MINUTE` (default 20). Send the secret as `X-API-Key: <secret>` or `Authorization: Bearer <secret>`. With no keys configured every `/ask` returns 503 instead of running open. `/health` needs no key.

| request | result |
|---|---|
| no or wrong key | 401 |
| over the per-key limit | 429 with `Retry-After` (seconds) |
| model provider failure (quota, timeout, 5xx) | 502 with the reason |
| `{"question": ...}` | JSON: `answer`, `mode`, `sources`, `disclaimer`, and `usage` (model, tokens, `cost_usd`, generation and total seconds) |
| `{"question": ..., "stream": true}` | Server-Sent Events: one `sources` event, then `token` events as the model writes, then `done` (mode, disclaimer, usage) or `error` |

```bash
curl -N localhost:8000/ask -H "X-API-Key: $KEY" -H 'content-type: application/json' \
  -d '{"question":"Does OPT authorization end if I transfer to another school?","stream":true}'
```

Every `/ask` writes one JSON line to stdout: key *name* (never the secret), status, retrieval mode, question length, number of sources, answer mode, model, tokens, `cost_usd`, latency, error. The question and answer text are deliberately not logged. `cost_usd` is `0.0` on the free Gemini tier (not a list price) and the Anthropic list price otherwise.

Limits, stated plainly: the rate limiter is in memory (one process; counts reset on restart; rejected keys do not use up anyone's allowance). All requests share one database connection. There is no per-key spend cap. Streaming through Gemini is real token streaming (`streamGenerateContent`), retried only before the first byte; Gemini's free tier is slow and flaky, and one live non-streaming call timed out (the gateway now returns 502 for that).

## MCP server

`app/mcp_server.py` uses the official Python SDK (`mcp` 2.3.0, `MCPServer`) and speaks stdio, so a client starts it as a subprocess. Tools:

| tool | what it does |
|---|---|
| `search_regulations(query, top_k=6)` | the retrieval pipeline above; returns passages with text, `citation`, a short `citation_id`, `kind` (REGULATION or COMMENTARY), url, score |
| `get_section(citation_id)` | the full indexed text of one section, e.g. `8 CFR 214.2(f)(10)`, in document order (cut at 20,000 characters, flagged `truncated`) |
| `ask(question)` | the full cited answer from the server's configured model, with sources and usage; answer is null if no model is configured |

Register it (replace the path with your checkout; the server reads the repo-root `.env` for the database and model settings):

```bash
claude mcp add --env PYTHONPATH=/ABSOLUTE/PATH/opt-compass/rag-service --transport stdio opt-compass \
  -- /ABSOLUTE/PATH/opt-compass/rag-service/.venv/bin/python -m app.mcp_server
```

or, for Claude Desktop, in its `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "opt-compass": {
      "command": "/ABSOLUTE/PATH/opt-compass/rag-service/.venv/bin/python",
      "args": ["-m", "app.mcp_server"],
      "env": { "PYTHONPATH": "/ABSOLUTE/PATH/opt-compass/rag-service" }
    }
  }
}
```

The database container must be running (`docker compose up -d db`). The first tool call loads the embedding model, which takes a few seconds.

**What has and has not been verified.** `python examples/mcp_client_demo.py` starts the server as a subprocess and calls every tool through the official MCP client over stdio; the output of that run (2026-11-01, real database, real Gemini free-tier call for `ask`) is saved in `rag-service/examples/mcp_transcript_2026-11-01.txt`. I have **not** run it inside Claude Desktop or Claude Code (the `claude` command is not installed on the machine this was built on), so the two registration snippets above are from the Claude Code docs and the SDK, not from a session I watched. Note also that in that transcript the top search result for "does OPT end if I transfer schools" is not the paragraph that answers it (that one is in `ask`'s sources); `search_regulations` is the same retrieval whose hit@1 is 0.82 on the dev set.

## Agent (LangGraph)

`app/agent.py`, run with `python -m app.agent "<question>"`. Graph: `plan` (one model call: in scope, legal advice, or out of scope, and the question split into up to 4 parts) then `research` (retrieval for each part, 3 passages each, at most 10 in total, no model call) then `synthesize` (one model call: a cited answer, plus up to 2 follow-up searches for parts the sources did not cover). If it asked for follow-ups, one more research and synthesize round runs, then it stops. Legal-advice requests (what *should I* do, *will I* be approved, is my employer breaking the law) and anything outside F-1 / OPT / STEM OPT end at `refuse` before any retrieval, with a pointer to a DSO or immigration attorney.

Step limit: every node counts as a step; at `max_steps` (default 8) the graph stops and returns what it has with status `step_limit`. A second guard, LangGraph's `recursion_limit`, is set from it. The longest live run used 5 steps, so the limit has only been triggered in unit tests with a fake model, never live.

### Agent vs single-shot (10 multi-part questions, 2026-11-01)

```bash
cd rag-service
LLM_PROVIDER=gemini LLM_MODEL=gemini-3.5-flash-lite .venv/bin/python -m eval.agent_eval run --out eval/results/agent-vs-single__gemini-3.5-flash-lite.jsonl
.venv/bin/python -m eval.agent_eval report eval/results/agent-vs-single__gemini-3.5-flash-lite.jsonl
```

Single-shot is the normal pipeline (top 6 passages for the whole question, one model call). The 10 questions have 21 parts in total; the questions and the substrings that count are in `eval/agent_questions.jsonl`. Both systems used `gemini-3.5-flash-lite` on the free tier.

| | single-shot | agent |
|---|---|---|
| parts whose answer-bearing text was in the passages (of 21) | 13 (14 with 10 passages) | 17 |
| parts whose key fact appears in the answer text (of 21) | 15 | 18 |
| questions with every part stated (of 10) | 5 | 7 |
| answers containing "The indexed sources do not answer this." | 2 | 2 |
| model calls per question (mean) | 1 | 2.3 |
| input / output tokens per question (mean) | 2,025 / 73 | 2,553 / 220 |
| median wall-clock time per question | 21.8 s | 52.1 s |
| faithfulness of the answers | **not measured** | **not measured** |

How to read it. The agent retrieved the needed text for 4 more parts than single-shot (3 more than single-shot with 10 passages), and the 10-passage control shows most of that comes from splitting the question, not from seeing more text. It costs about 2.3 times the model calls, three times the output tokens and, here, 2.4 times the time (the free tier forces a 6.5 s pause between calls, so absolute times are not comparable to a paid API). The gain is small and the sample is smaller: 10 questions, 21 parts, no confidence interval, questions I wrote while looking at the corpus. "Stated" is a substring check, so a wrong answer that contains the right number would pass; the judge step (`python -m eval.agent_eval judge <run>`) that would measure faithfulness for both systems needs about 20 judge calls and was not run because the free tier's daily limit was nearly used up. Until it is, nothing here says the agent's answers are as faithful as single-shot's.

Where both still fail:

- **a06 (cap-gap end date).** Both answers say the extension lasts until October 1. The current regulation says April 1 and neither system retrieved that paragraph, so the source labels that fixed q025 had nothing to prefer. This is the commentary-against-regulation problem again, and a wrong answer that sounds sure.
- **a08 (STEM OPT filing deadline and appeal).** Single-shot abstained on both parts. The agent got the 60-day deadline and still found nothing on the appeal, after a follow-up search (the "may not appeal" paragraph is in the corpus).
- **Where the retrieval gain came from.** The agent found text that single-shot did not for a03, a04 and a10 (one part each) and a05 (both parts). It went the other way on a07 (1 of 2 parts' text against single-shot's 2), though its answer still states the number. Net: 17 against 13.

### Refusal path (8 probes)

3 legal-advice, 3 out-of-scope and 2 in-scope controls (one phrased with "I am on post-completion OPT", to catch over-refusal): 8 of 8 classified as expected, in `eval/agent_refusals.jsonl`. The probes are mine, plain and non-adversarial; there is no prompt-injection or paraphrase test, so this shows the path works, not that it is robust. Classification is the model's call: a malformed plan falls back to treating the question as in scope, because the answer step is still grounded in the sources.

## Sources and one deliberate limit

uscis.gov and ice.gov return HTTP 403 to scripted requests. This project does not try to get around that. Pages from those sites can be added by saving them yourself into `data/manual/` (see the README there). Regulation citations are pinned to the top-level paragraph, e.g. `8 CFR 214.2(f)(10) Practical training`; the deeper outline levels ((ii), (E), ...) cannot be recovered reliably from the flat eCFR XML, so they stay in the quoted text and are never guessed into a citation.

## Run it

```bash
docker compose up -d db                 # PostgreSQL 16 + pgvector on :5433
cd rag-service
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m app.ingest          # first run downloads the embedding model; ~10 min on CPU
.venv/bin/python -m eval.run_eval       # retrieval metrics (--validate checks the eval phrases exist)
.venv/bin/uvicorn app.main:app --port 8000
# the gateway refuses every request unless API_KEYS is set (see the next section)
curl -s localhost:8000/ask -H 'content-type: application/json' -H "X-API-Key: $KEY" \
  -d '{"question":"How many days of unemployment are allowed on post-completion OPT?"}'
.venv/bin/python -m pytest              # unit tests, no database or network needed
```

Answer evaluation (needs a model key in the repo-root `.env`; the free Gemini tier works but is rate limited, see `.env.example`):

```bash
.venv/bin/python -m eval.answer_eval models     # model IDs your key can call (do not guess IDs)
.venv/bin/python -m eval.answer_eval estimate   # token volume and list-price cost
.venv/bin/python -m eval.answer_eval validate   # checks the unanswerable questions really are absent from the corpus
LLM_MODEL=<generator> JUDGE_MODEL=<judge> .venv/bin/python -m eval.answer_eval run --out eval/results/<name>.jsonl
.venv/bin/python -m eval.answer_eval report eval/results/<name>.jsonl
```

Tracing is optional: `observability/make-env.sh` writes a gitignored `observability/.env`, then `docker compose -f observability/docker-compose.langfuse.yml --env-file observability/.env up -d` starts a self-hosted Langfuse (about 1.7 GB of RAM) and the app traces when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set.

Configuration is environment variables; see `.env.example`. Never commit `.env`.

## Planned

| milestone | scope | status |
|---|---|---|
| M1 | retrieval slice and retrieval eval | built and tested |
| M1.5 | answer-quality eval, judge validation, Langfuse tracing, then retrieval and prompt fixes | built and tested; fixes partly re-evaluated |
| M2 | FastAPI gateway (SSE streaming, API-key auth, rate limit, cost per request), MCP server, LangGraph agent with refusal path | built and tested; agent vs single-shot measured on 10 questions, faithfulness judge step not yet run |
| M3 | Dockerfiles, GitHub Actions CI with a retrieval-eval gate, AWS deployment | planned |
