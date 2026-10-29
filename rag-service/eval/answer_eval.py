"""Answer-quality eval (separate from the retrieval eval in run_eval.py).

  python -m eval.answer_eval estimate          # approximate token cost, no API calls
  python -m eval.answer_eval run --out eval/results/<name>.jsonl
  python -m eval.answer_eval report eval/results/<name>.jsonl
  python -m eval.answer_eval sample eval/results/<name>.jsonl     # claims to hand-check
  python -m eval.answer_eval agreement eval/results/<name>.jsonl eval/manual/labels.jsonl

`run` calls the Anthropic API (LLM_MODEL generates, JUDGE_MODEL judges) and costs money.
"""
import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

from app.config import load_settings
from app.db import connect
from app.generate import answer as generate_answer
from app.generate import format_sources
from app.pricing import PRICES
from app.retrieve import search
from eval.answer_metrics import (
    Claim,
    ScoredAnswer,
    Verdict,
    aggregate,
    cohens_kappa,
    gate,
    is_abstention,
    split_claims,
    valid_citations,
)
from eval.judge import JudgeUsage, build_prompt, make_llm_judge

HERE = Path(__file__).parent
TOP_K = 6
MODE = "rerank"


def load_questions() -> list[dict]:
    items = []
    for line in (HERE / "questions.jsonl").read_text().splitlines():
        if line.strip():
            items.append({**json.loads(line), "kind": "answerable"})
    for line in (HERE / "unanswerable.jsonl").read_text().splitlines():
        if line.strip():
            items.append({**json.loads(line), "kind": "unanswerable"})
    for n, item in enumerate(items, start=1):
        item["id"] = f"q{n:03d}"
    return items


def validate_unanswerable(conn) -> int:
    bad = 0
    for q in (q for q in load_questions() if q["kind"] == "unanswerable"):
        for phrase in q["absent_phrases"]:
            n = conn.execute("SELECT count(*) FROM chunks WHERE lower(content) LIKE %s", (f"%{phrase.lower()}%",)).fetchone()[0]
            if n:
                print(f"NOT absent from corpus ({n} chunks): {phrase!r} -> {q['question']}")
                bad += 1
    print(f"validated unanswerable questions, {bad} phrase(s) actually present")
    return bad


def estimate(conn) -> None:
    """Approximate cost per candidate model. Tokens are estimated as chars/3.5 (conservative)."""
    questions = load_questions()
    gen_in = judge_in = 0
    for q in questions:
        hits = search(conn, q["question"], TOP_K, MODE)
        prompt_chars = len(format_sources(hits)) + len(q["question"]) + 700  # system prompt allowance
        gen_in += prompt_chars / 3.5
        judge_in += (prompt_chars + 900) / 3.5  # + claims list and judge instructions
    n = len(questions)
    gen_out, judge_out = 250 * n, 350 * n
    print(f"{n} questions: ~{gen_in:,.0f} generation input tokens, ~{gen_out:,} output (assumed 250/answer);")
    print(f"judge ~{judge_in:,.0f} input, ~{judge_out:,} output (assumed 350/answer). Thinking models add output tokens.")
    print(f"\n{'model':<20}{'generate $':>12}{'judge $':>10}   (list price, per full run)")
    for model, (pin, pout) in PRICES.items():
        g = (gen_in * pin + gen_out * pout) / 1e6
        j = (judge_in * pin + judge_out * pout) / 1e6
        print(f"{model:<20}{g:>12.3f}{j:>10.3f}")


def run(out: Path, resume: bool) -> None:
    settings = load_settings()
    if not settings.anthropic_api_key:
        sys.exit("ANTHROPIC_API_KEY is not set; refusing to run (no mocked results).")
    if not settings.judge_model:
        sys.exit("JUDGE_MODEL is not set.")
    conn = connect()
    judge_usage = JudgeUsage()
    judge = make_llm_judge(judge_usage)
    done = set()
    if resume and out.exists():
        done = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a" if resume else "w") as f:
        for q in load_questions():
            if q["id"] in done:
                continue
            hits = search(conn, q["question"], TOP_K, MODE)
            ans = generate_answer(q["question"], hits)
            passages = [h.content for h in hits]
            claims, verdicts = [], []
            abstained = is_abstention(ans.text)
            if ans.text and not abstained:
                claims = split_claims(ans.text)
                if claims:
                    verdicts = judge(q["question"], claims, passages)
            record = {
                "id": q["id"],
                "kind": q["kind"],
                "question": q["question"],
                "answer": asdict(ans),
                "abstained": abstained,
                "passages": [{"n": i, "citation": h.citation, "content": h.content} for i, h in enumerate(hits, 1)],
                "claims": [asdict(c) for c in claims],
                "verdicts": [asdict(v) for v in verdicts],
                "config": {"llm_model": settings.llm_model, "llm_effort": settings.llm_effort,
                           "judge_model": settings.judge_model, "top_k": TOP_K, "mode": MODE},
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(f"{q['id']} {q['kind']:<12} abstained={abstained} claims={len(claims)} cost={ans.cost_usd}")
    print(f"\njudge: {asdict(judge_usage)}")


def load_run(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def to_scored(record: dict) -> ScoredAnswer:
    return ScoredAnswer(
        kind=record["kind"],
        abstained=record["abstained"],
        claims=[Claim(**c) for c in record["claims"]],
        verdicts=[Verdict(**v) for v in record["verdicts"]],
        n_passages=len(record["passages"]),
    )


def report(path: Path) -> dict:
    records = load_run(path)
    metrics = aggregate([to_scored(r) for r in records])
    answers = [r["answer"] for r in records]
    lat = sorted(a["latency_s"] for a in answers if a["latency_s"])
    costs = [a["cost_usd"] for a in answers]
    metrics.update(
        {
            "generation_cost_usd_total": None if any(c is None for c in costs) else sum(costs),
            "generation_cost_usd_per_answer": None if any(c is None for c in costs) else sum(costs) / len(costs),
            "latency_s_median": statistics.median(lat) if lat else None,
            "latency_s_p95": lat[int(0.95 * (len(lat) - 1))] if lat else None,
            "input_tokens_mean": statistics.mean(a["input_tokens"] for a in answers),
            "output_tokens_mean": statistics.mean(a["output_tokens"] for a in answers),
            "modes": {m: sum(1 for a in answers if a["mode"] == m) for m in {a["mode"] for a in answers}},
            "config": records[0]["config"],
        }
    )
    for key, value in metrics.items():
        print(f"{key:<32}{value if not isinstance(value, float) else round(value, 3)}")
    failures = gate(metrics)
    print("\nGATE:", "PASS" if not failures else "FAIL\n  " + "\n  ".join(failures))
    return metrics


def sample(path: Path, n_answers: int = 12) -> None:
    """Pick answers to hand-check: spread across answerable questions plus a few that abstained/answered unanswerable."""
    records = load_run(path)
    pool = [r for r in records if r["claims"]]
    step = max(1, len(pool) // n_answers)
    picked = pool[::step][:n_answers]
    print(f"# Manual review sheet ({len(picked)} answers). For each claim, write a line to eval/manual/labels.jsonl:")
    print('# {"id": "q001", "claim": 1, "supported_by_any": true, "supported_by_cited": true, "labeler": "<name>"}\n')
    for r in picked:
        print(f"## {r['id']}  {r['question']}")
        for c in r["passages"]:
            print(f"[{c['n']}] {c['citation']}\n{c['content']}\n")
        for i, c in enumerate(r["claims"], start=1):
            print(f"  claim {i} cites {c['cited']}: {c['text']}")
        print()


def agreement(run_path: Path, labels_path: Path) -> None:
    records = {r["id"]: r for r in load_run(run_path)}
    labels = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
    pairs = {"supported_by_any": ([], []), "supported_by_cited": ([], [])}
    answers = set()
    for lab in labels:
        verdicts = records[lab["id"]]["verdicts"]
        v = verdicts[lab["claim"] - 1]
        answers.add(lab["id"])
        for key, (human, judge) in pairs.items():
            human.append(bool(lab[key]))
            judge.append(bool(v[key]))
    print(f"{len(labels)} hand-labelled claims across {len(answers)} answers")
    for key, (human, judge) in pairs.items():
        raw = sum(h == j for h, j in zip(human, judge)) / len(human)
        kappa = cohens_kappa(human, judge)
        disagreements = [(l["id"], l["claim"]) for l, h, j in zip(labels, human, judge) if h != j]
        print(f"{key}: raw agreement {raw:.2f}, Cohen's kappa {kappa if kappa is None else round(kappa, 2)}, disagreements {disagreements}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("estimate")
    sub.add_parser("validate")
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--resume", action="store_true")
    for name in ("report", "sample"):
        sub.add_parser(name).add_argument("path", type=Path)
    a = sub.add_parser("agreement")
    a.add_argument("path", type=Path)
    a.add_argument("labels", type=Path)
    args = parser.parse_args()
    if args.cmd == "estimate":
        estimate(connect())
    elif args.cmd == "validate":
        sys.exit(1 if validate_unanswerable(connect()) else 0)
    elif args.cmd == "run":
        run(args.out, args.resume)
    elif args.cmd == "report":
        sys.exit(1 if gate(report(args.path)) else 0)
    elif args.cmd == "sample":
        sample(args.path)
    elif args.cmd == "agreement":
        agreement(args.path, args.labels)


if __name__ == "__main__":
    main()
