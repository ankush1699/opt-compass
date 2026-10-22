# OPT Compass

Cited answers to F-1 / OPT / STEM OPT questions, grounded in the primary sources rather than forum posts:
the current text of **8 CFR 214.2(f)** (eCFR API) and the **STEM OPT final rule, 81 FR 13040** (Federal Register API).

> General information from public regulations, not legal advice. Confirm anything important with your school's DSO or an immigration attorney.

**Status: milestone 1 (retrieval slice).** Only `rag-service/` exists. The Spring Boot gateway, scheduled ingestion, frontend and AWS deployment are later milestones and are not built yet.

## What works today

- **Ingestion** (`python -m app.ingest`): fetches both sources, chunks them, embeds them and upserts into PostgreSQL + pgvector. Re-running is idempotent (content-hash dedupe).
- **Retrieval**: dense (pgvector cosine), keyword (Postgres full text), hybrid (reciprocal rank fusion) and `rerank` (hybrid pool re-scored by a cross-encoder). Regulation text is weighted above Federal Register commentary.
- **API** (FastAPI): `POST /ask` returns the sources with citations and snippets. With `ANTHROPIC_API_KEY` set it also returns a grounded answer whose sentences cite `[n]`; without a key it runs in retrieval-only mode.
- **Evaluation** (`python -m eval.run_eval`): 28 questions, each with the exact phrase the answer-bearing passage must contain.

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

**Caveats, stated plainly.** 28 questions is a development set, and the chunk size and source weights were tuned on it, so treat the numbers as a baseline and not a generalisation claim. The generation step (grounded, cited answers) is implemented but has **not** been evaluated for faithfulness yet; that is the next piece of M1 work.

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

Configuration is environment variables; see `.env.example`. Never commit `.env`.

## Planned

| milestone | scope |
|---|---|
| M1 (in progress) | retrieval slice above, plus a faithfulness check of generated answers |
| M2 | Spring Boot gateway: JWT auth, rate limiting, per-user cost guard, calls rag-service |
| M3 | AWS deployment (Terraform, GitHub Actions CD) and scheduled ingestion |
