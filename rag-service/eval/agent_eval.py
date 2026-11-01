"""Agent vs single-shot on multi-part questions, plus the agent's refusal path.

  python -m eval.agent_eval validate                       # every answer-bearing phrase exists in the corpus (no model calls)
  python -m eval.agent_eval run --out eval/results/<name>.jsonl [--resume]   # both systems, 10 questions + 8 refusal probes
  python -m eval.agent_eval judge <run.jsonl>              # LLM-judge faithfulness for both systems (needs JUDGE_MODEL; resumable)
  python -m eval.agent_eval report <run.jsonl>

Each multi-part question lists its parts. For each part the report checks, without any model:
  retrieved - some passage the system was given contains every `must_contain` phrase
  stated    - the answer text contains one of the `expect_any` strings (a crude substring check, so it can be fooled
              by a wrong answer that happens to contain the string; faithfulness is what the judge step is for)
Single-shot is the normal pipeline: top 6 passages for the whole question, one model call. The agent splits the question,
retrieves 3 passages per part (up to 10 in total) and writes one answer, so it sees different (not fewer) passages.
"""
import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

from app.agent import run_agent
from app.config import load_settings
from app.db import connect
from app.generate import answer as generate_answer
from app.llm import has_credentials
from app.retrieve import search
from eval.answer_metrics import aggregate, is_abstention, split_claims, ScoredAnswer, Claim, Verdict
from eval.judge import JudgeUsage, make_llm_judge

HERE = Path(__file__).parent


def load_items() -> list[dict]:
    items = []
    for n, line in enumerate((HERE / "agent_questions.jsonl").read_text().splitlines(), start=1):
        if line.strip():
            items.append({**json.loads(line), "id": f"a{n:02d}", "kind": "multi"})
    for n, line in enumerate((HERE / "agent_refusals.jsonl").read_text().splitlines(), start=1):
        if line.strip():
            items.append({**json.loads(line), "id": f"r{n:02d}", "kind": "refusal"})
    return items


def load_run(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def part_retrieved(passages: list[str], part: dict) -> bool:
    return any(all(p in text.lower() for p in part["must_contain"]) for text in passages)


def part_stated(answer: str | None, part: dict) -> bool:
    return bool(answer) and any(s in answer.lower() for s in part["expect_any"])


def validate(conn) -> int:
    missing = 0
    for item in (i for i in load_items() if i["kind"] == "multi"):
        for part in item["parts"]:
            for phrase in part["must_contain"]:
                n = conn.execute("SELECT count(*) FROM chunks WHERE lower(content) LIKE %s", (f"%{phrase}%",)).fetchone()[0]
                if not n:
                    print(f"MISSING from corpus: {phrase!r} ({item['question']})")
                    missing += 1
    print(f"validated, {missing} missing phrase(s)")
    return missing


def run(out: Path, resume: bool) -> None:
    settings = load_settings()
    if not has_credentials(settings.llm_provider, settings):
        sys.exit(f"No API key for provider {settings.llm_provider!r}; refusing to run (no mocked results).")
    conn = connect()
    done = {r["id"] for r in load_run(out)} if resume and out.exists() else set()
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {"llm_provider": settings.llm_provider, "llm_model": settings.llm_model}
    with out.open("a" if resume else "w") as f:
        for item in load_items():
            if item["id"] in done:
                continue
            record = {"id": item["id"], "kind": item["kind"], "question": item["question"], "config": config}
            if item["kind"] == "multi":
                record["parts"] = item["parts"]
                hits = search(conn, item["question"], 6, "rerank")
                started = time.perf_counter()
                single = generate_answer(item["question"], hits)
                record["single"] = {
                    "answer": single.text, "mode": single.mode, "llm_calls": 1, "input_tokens": single.input_tokens,
                    "output_tokens": single.output_tokens, "latency_s": time.perf_counter() - started,
                    "passages": [{"citation": h.citation, "content": h.content} for h in hits],
                }
            else:
                record["expected_intent"] = item["expected_intent"]
            started = time.perf_counter()
            result = run_agent(item["question"], conn)
            record["agent"] = {
                "answer": result.answer, "status": result.status, "intent": result.intent, "sub_questions": result.sub_questions,
                "steps": result.steps, "llm_calls": result.llm_calls, "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens, "cost_usd": result.cost_usd, "trace": result.trace,
                "latency_s": time.perf_counter() - started,
                "passages": [{"citation": h.citation, "content": h.content} for h in result.sources],
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(f"{item['id']} {item['kind']:<7} agent: {result.status}/{result.intent} {result.llm_calls} calls {result.steps} steps")


def judge_run(path: Path) -> None:
    """Fills in per-claim verdicts for both systems; skips what is already judged, so a quota stop can be resumed."""
    settings = load_settings()
    if not settings.judge_model:
        sys.exit("JUDGE_MODEL is not set.")
    judge = make_llm_judge(JudgeUsage())
    records = load_run(path)
    for record in records:
        if record["kind"] != "multi":
            continue
        for system in ("single", "agent"):
            side = record[system]
            if "verdicts" in side or not side["answer"] or is_abstention(side["answer"]):
                continue
            claims = split_claims(side["answer"])
            side["claims"] = [asdict(c) for c in claims]
            side["verdicts"] = [asdict(v) for v in judge(record["question"], claims, [p["content"] for p in side["passages"]])] if claims else []
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            print(f"judged {record['id']} {system}: {len(claims)} claims")
    print("judge complete")


def _scored(side: dict) -> ScoredAnswer:
    return ScoredAnswer(
        kind="answerable",
        abstained=is_abstention(side["answer"]),
        claims=[Claim(**c) for c in side.get("claims", [])],
        verdicts=[Verdict(**v) for v in side.get("verdicts", [])],
        n_passages=len(side["passages"]),
    )


def report(path: Path) -> dict:
    records = load_run(path)
    multi = [r for r in records if r["kind"] == "multi"]
    out: dict = {"questions": len(multi), "parts": sum(len(r["parts"]) for r in multi)}
    for system in ("single", "agent"):
        parts = [(r, p) for r in multi for p in r["parts"]]
        sides = [r[system] for r in multi]
        row = {
            "parts_retrieved": sum(part_retrieved([x["content"] for x in r[system]["passages"]], p) for r, p in parts),
            "parts_stated": sum(part_stated(r[system]["answer"], p) for r, p in parts),
            "questions_all_parts_stated": sum(all(part_stated(r[system]["answer"], p) for p in r["parts"]) for r in multi),
            "contains_abstain_phrase": sum("do not answer this" in (s["answer"] or "").lower() for s in sides),
            "llm_calls_mean": statistics.mean(s["llm_calls"] for s in sides),
            "input_tokens_mean": statistics.mean(s["input_tokens"] for s in sides),
            "output_tokens_mean": statistics.mean(s["output_tokens"] for s in sides),
            "latency_s_median": statistics.median(s["latency_s"] for s in sides),
        }
        judged = [r[system] for r in multi if "verdicts" in r[system]]
        if judged:
            m = aggregate([_scored(s) for s in judged])
            row.update({"judged_answers": len(judged), "faithfulness": m.get("faithfulness"), "citation_precision": m.get("citation_precision")})
        out[system] = row
    refusals = [r for r in records if r["kind"] == "refusal"]
    if refusals:
        out["refusal_probes"] = [
            {"id": r["id"], "expected": r["expected_intent"], "got": r["agent"]["intent"], "status": r["agent"]["status"], "ok": r["agent"]["intent"] == r["expected_intent"]}
            for r in refusals
        ]
    out["agent_steps_max"] = max((r["agent"]["steps"] for r in records), default=0)
    print(json.dumps(out, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--resume", action="store_true")
    for name in ("judge", "report"):
        sub.add_parser(name).add_argument("path", type=Path)
    args = parser.parse_args()
    if args.cmd == "validate":
        sys.exit(1 if validate(connect()) else 0)
    elif args.cmd == "run":
        run(args.out, args.resume)
    elif args.cmd == "judge":
        judge_run(args.path)
    elif args.cmd == "report":
        report(args.path)


if __name__ == "__main__":
    main()
