"""Retrieval evaluation.

A question is a hit at k when ONE of the top-k retrieved chunks contains every `must_contain`
phrase (case-insensitive), i.e. the answer-bearing passage was retrieved.

Two strictness levels are reported:
  any      - the passage may come from any source (regulation or Federal Register commentary)
  primary  - questions flagged `in_regulation` must be answered by an eCFR chunk, the current law

Usage: python -m eval.run_eval [--validate] [--heldout]   (--heldout: questions_heldout.jsonl, written after the
retrieval settings were tuned on questions.jsonl and never used to tune anything)   (--validate checks every phrase exists in the corpus)
"""
import json
import sys
from pathlib import Path

from app.db import connect
from app.retrieve import search

QUESTIONS = Path(__file__).with_name("questions.jsonl")
HELDOUT = Path(__file__).with_name("questions_heldout.jsonl")
KS = (1, 3, 6)
MODES = ("dense", "keyword", "hybrid", "rerank")


def load() -> list[dict]:
    path = HELDOUT if "--heldout" in sys.argv else QUESTIONS
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(conn, questions: list[dict]) -> int:
    missing = 0
    for q in questions:
        for phrase in q["must_contain"]:
            n = conn.execute("SELECT count(*) FROM chunks WHERE lower(content) LIKE %s", (f"%{phrase}%",)).fetchone()[0]
            if n == 0:
                print(f"MISSING from corpus: {phrase!r}  ({q['question']})")
                missing += 1
    print(f"validated {len(questions)} questions, {missing} missing phrases")
    return missing


def is_hit(hit, q: dict, primary: bool) -> bool:
    if primary and q["in_regulation"] and hit.source != "eCFR":
        return False
    return all(p in hit.content.lower() for p in q["must_contain"])


def evaluate(conn, questions: list[dict], mode: str) -> dict:
    result: dict = {}
    for level, primary in (("any", False), ("primary", True)):
        ranks = []
        for q in questions:
            hits = search(conn, q["question"], max(KS), mode)
            ranks.append(next((i for i, h in enumerate(hits, start=1) if is_hit(h, q, primary)), None))
        for k in KS:
            result[f"{level} hit@{k}"] = sum(1 for r in ranks if r and r <= k) / len(ranks)
        result[f"{level} mrr"] = sum(1 / r for r in ranks if r) / len(ranks)
        result[f"{level} misses"] = [q["question"] for q, r in zip(questions, ranks) if r is None]
    return result


def main() -> None:
    conn = connect()
    questions = load()
    if "--validate" in sys.argv:
        sys.exit(1 if validate(conn, questions) else 0)
    report = {mode: evaluate(conn, questions, mode) for mode in MODES}
    print(f"{len(questions)} questions\n")
    for level in ("any", "primary"):
        print(f"[{level}]  " + f"{'mode':<9}" + "".join(f"hit@{k:<5}" for k in KS) + "mrr")
        for mode, r in report.items():
            print(f"          {mode:<9}" + "".join(f"{r[f'{level} hit@{k}']:<10.2f}" for k in KS) + f"{r[f'{level} mrr']:.2f}")
        print()
    if "--misses" in sys.argv:
        for mode, r in report.items():
            for level in ("any", "primary"):
                if r[f"{level} misses"]:
                    print(f"{mode} / {level} misses:")
                    for m in r[f"{level} misses"]:
                        print("  -", m)


if __name__ == "__main__":
    main()
