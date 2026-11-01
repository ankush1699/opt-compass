"""Does the judge catch a wrong claim? Take real answers from a run, corrupt some claims, leave others as controls.

A judge that says "supported" to everything scores 1.0 faithfulness on any answer set, so agreement on correct
answers alone proves nothing. This runs the REAL judge on known-wrong claims. Cost: one judge call per case.

    python -m eval.judge_probe eval/results/<run>.jsonl --out eval/results/judge_probe.json
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from eval.answer_eval import load_run
from eval.answer_metrics import Claim
from eval.judge import JudgeUsage, make_llm_judge

# (record id, claim number, new text or None to keep, new citations or None to keep, kind)
# kind: "wrong" = the text is false per the passages; "miscited" = true text, wrong citation; "control" = untouched.
CASES = [
    ("q001", 1, "During post-completion OPT, students may not accrue an aggregate of more than 60 days of unemployment .", None, "wrong"),
    ("q001", 2, None, None, "control"),
    ("q003", 1, "You may properly file your Form I-765 for post-completion OPT up to 30 days prior to your program end date .", None, "wrong"),
    ("q005", 1, "No, a student does not need a recommendation from their DSO in order to apply for OPT .", None, "wrong"),
    ("q005", 2, None, None, "control"),
    ("q012", 1, "Students on STEM OPT extensions are required to report to their DSOs every twelve months to confirm the validity of their SEVIS information .", None, "wrong"),
    ("q016", 1, "Yes, an employer may replace U.S. workers with STEM OPT students once more than 120 days have passed .", None, "wrong"),
    ("q016", 2, None, None, "control"),
    ("q018", 1, None, None, "control"),
    ("q018", 2, "This list is maintained by USCIS and published at uscis.gov/stem-list .", None, "wrong"),
    ("q007", 1, None, [4], "miscited"),
]


def summarise(rows: list[dict], judge_model: str) -> dict:
    wrong = [r for r in rows if r["kind"] == "wrong"]
    controls = [r for r in rows if r["kind"] == "control"]
    miscited = [r for r in rows if r["kind"] == "miscited"]
    return {
        "judge_model": judge_model,
        "cases_run": len(rows),
        "cases_planned": len(CASES),
        "wrong_claims": len(wrong),
        "wrong_caught": sum(not r["supported_by_any"] for r in wrong),
        "controls": len(controls),
        "controls_wrongly_flagged": sum(not r["supported_by_any"] for r in controls),
        "miscited": len(miscited),
        "miscited_caught": sum(not r["supported_by_cited"] for r in miscited),
        "rows": rows,
    }


def probe(run_path: Path, out: Path, judge=None) -> dict:
    """Saves after every case, so a quota error mid-run keeps what was measured (cases_run < cases_planned says so)."""
    records = {r["id"]: r for r in load_run(run_path)}
    judge = judge or make_llm_judge(JudgeUsage())
    judge_model = records[CASES[0][0]]["config"]["judge_model"]
    rows: list[dict] = []
    for rid, number, text, cited, kind in CASES:
        record = records[rid]
        original = record["claims"][number - 1]
        claim = Claim(text or original["text"], cited if cited is not None else original["cited"])
        verdict = judge(record["question"], [claim], [p["content"] for p in record["passages"]])[0]
        rows.append({"id": rid, "claim": number, "kind": kind, "text": claim.text, "cited": claim.cited, **asdict(verdict)})
        out.write_text(json.dumps(summarise(rows, judge_model), indent=2))
        print(f"{rid} c{number} {kind:<8} any={verdict.supported_by_any!s:<5} cited={verdict.supported_by_cited!s:<5} {verdict.reason[:100]}")
    summary = summarise(rows, judge_model)
    print({k: v for k, v in summary.items() if k != "rows"})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    probe(args.run, args.out)
