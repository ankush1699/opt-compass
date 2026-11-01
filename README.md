# OPT Compass

Cited answers to F-1 / OPT / STEM OPT questions, grounded in the primary sources rather than forum posts:
the current text of **8 CFR 214.2(f)** (eCFR API) and the **STEM OPT final rule, 81 FR 13040** (Federal Register API).

> General information from public regulations, not legal advice. Confirm anything important with your school's DSO or an immigration attorney.

**Status: milestone 1.5 (answer-quality evaluation and tracing).** Only `rag-service/` and `observability/` exist. The API gateway, MCP server, agent, CI and AWS deployment are later milestones and are not built yet.

## Feature status

| feature | status |
|---|---|
| Ingestion from eCFR and Federal Register APIs | built and tested |
| Retrieval (dense, keyword, hybrid, rerank) | built and tested; retrieval metrics below |
| `POST /ask` with cited answers | built and tested |
| Answer-quality eval (faithfulness, citation accuracy, abstention) | built and tested; one live run, results below |
| Judge validation by hand-labelling | built and tested; 18 claims labelled, results below |
| Judge sensitivity probe (`eval/judge_probe.py`) | built, not evaluated: the live run hit the free-tier daily limit and has not been completed |
| Langfuse tracing (self-hosted) | built and tested for retrieval and rerank spans; generation token and cost recording verified only against a fake model reply, not a live traced run |
| FastAPI gateway, MCP server, LangGraph agent | planned (M2) |
| Dockerfiles, CI, AWS deployment | planned (M3) |

## What works today

- **Ingestion** (`python -m app.ingest`): fetches both sources, chunks them, embeds them and upserts into PostgreSQL + pgvector. Re-running is idempotent (content-hash dedupe).
- **Retrieval**: dense (pgvector cosine), keyword (Postgres full text), hybrid (reciprocal rank fusion) and `rerank` (hybrid pool re-scored by a cross-encoder). Regulation text is weighted above Federal Register commentary.
- **API** (FastAPI): `POST /ask` returns the sources with citations and snippets. With a model configured (`LLM_PROVIDER`, `LLM_MODEL`, and the matching key) it also returns a grounded answer whose sentences cite `[n]`, or the sentence "The indexed sources do not answer this."; without one it runs in retrieval-only mode.
- **Retrieval evaluation** (`python -m eval.run_eval`): 28 questions, each with the exact phrase the answer-bearing passage must contain.
- **Answer evaluation** (`python -m eval.answer_eval`): 28 answerable plus 8 unanswerable questions, graded by an LLM judge claim by claim.

### Retrieval results (28-question dev set, `BAAI/bge-base-en-v1.5`)

| mode | any-source hit@1 | hit@6 | primary-source hit@1 | hit@6 |
|---|---|---|---|---|
| dense | 0.64 | 0.93 | 0.61 | 0.82 |
| keyword | 0.11 | 0.61 | 0.11 | 0.54 |
| hybrid | 0.54 | 0.86 | 0.54 | 0.75 |
| **rerank (default)** | **0.68** | **0.93** | **0.64** | **0.79** |

*any-source*: a passage from the regulation or the commentary counts. *primary-source*: questions the regulation answers must be answered by an eCFR chunk.

What the experiments showed (each change was measured on the same set):

| change | effect |
|---|---|
| 700-char regulation chunks instead of 1,800 | dense hit@6 0.57 → 0.64 (one rule per chunk) |
| down-weight commentary vs. regulation | dense hit@6 0.64 → 0.71; the 425 preamble chunks otherwise crowd out the 125 regulation chunks |
| `bge-small` (384d) → `bge-base` (768d) | any-source dense hit@6 0.86 → 0.93 |
| hybrid fusion | *worse* than dense alone; the keyword leg adds noise on a corpus this small, so it is kept only as an option |

**Caveats, stated plainly.** 28 questions is a development set, and the chunk size and source weights were tuned on it, so treat the numbers as a baseline and not a generalisation claim.

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

### How far to trust the judge

| check | result |
|---|---|
| 18 claims from 12 answers labelled by hand | `supported_by_any`: raw agreement 0.94, Cohen's kappa 0.0; `supported_by_cited`: raw agreement 0.94, kappa 0.77 |
| judge sensitivity probe (known-wrong claims) | **not completed**, see below |

Read this table carefully. The judge marked all 45 claims as supported, so on that axis there is nothing for kappa to measure: kappa 0.0 does not mean "random", it means the judge never varied, and 0.94 agreement is simply 17 of 18 labels matching. The one disagreement (q025, claim 2) is a case the judge missed, and it is the only evidence in this repo that the judge can be too lenient. The labeller was Claude, not a human, and had seen the judge's overall result before labelling, so the labels are neither blind nor independent of model family. The labels are in `rag-service/eval/manual/labels.jsonl` and anyone can re-check them with `python -m eval.answer_eval sample <run>` and `agreement <run> <labels>`.

`eval/judge_probe.py` corrupts real claims (wrong number, flipped answer, invented source, wrong citation) and checks whether the real judge flags them, with untouched claims as controls. Its logic is unit-tested against a lenient and a strict fake judge. The live run stopped after the free tier's daily request limit was reached, so there is **no recorded sensitivity number yet**.

## Known failures

- **q008 (transfer ends OPT?): retrieval miss.** The right section, `8 CFR 214.2(f)(8)`, ranked first, but the chunk holding the sentence about automatic termination was not in the 6 passages. The model abstained, which is the correct behaviour given what it saw. This is a retrieval failure, and it is the same kind of miss the hit@6 numbers above leave room for.
- **q020 (travel on post-completion OPT?): generation failure.** Passage 2 contained the answer and the model still said the sources do not answer. This is the small generator being too cautious.
- **Uncited claims (q014, q023, q027).** Three claims carry no `[n]`, and the judge correctly reported them as supported but not by a citation. One of them is the lead-in sentence of a list.
- **Old commentary against current regulation (q025).** The answer says the cap-gap extension lasts "until April 1 ... or until the beginning of the new fiscal year (October 1)". The current regulation says April 1; the 2016 preamble describes the older October 1 rule. The model merged them as alternatives and the judge accepted it. The corpus contains both and nothing marks the commentary as superseded.
- **The generator writes very short answers** (mean about 54 output tokens). Short answers contain fewer claims, which makes a faithfulness score of 1.00 easier to reach. It is not evidence that a more verbose model would score the same.
- **Judge and generator are both Google models**, so shared blind spots and self-preference are possible, and the judge never marked any of the 45 claims unsupported.
- **Single run, no variance estimate.** One configuration, one pass, 36 questions. There are no confidence intervals and the retrieval and answer questions overlap, so these are development-set numbers.
- **Free-tier data use.** Google's pricing page lists free-tier content as used to improve its products. Only public regulation text and synthetic questions were sent.
- **Free-tier operations.** The first attempt failed on `503 high demand` for the first two models chosen, and the judge allows only 20 requests per day, which is why the run spans two sessions.

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
curl -s localhost:8000/ask -H 'content-type: application/json' \
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
| M1.5 | answer-quality eval, judge validation, Langfuse tracing | built and tested, judge sensitivity probe pending |
| M2 | FastAPI gateway (SSE streaming, API-key auth, rate limit, cost per request), MCP server, LangGraph agent with refusal path | planned |
| M3 | Dockerfiles, GitHub Actions CI with a retrieval-eval gate, AWS deployment | planned |
