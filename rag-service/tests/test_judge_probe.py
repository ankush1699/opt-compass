import json

from eval.answer_metrics import Verdict
from eval.judge_probe import CASES, probe


def run_file(tmp_path):
    ids = {c[0] for c in CASES}
    records = [
        {
            "id": rid,
            "question": "q",
            "claims": [{"text": f"orig {rid} {n}", "cited": [1]} for n in (1, 2)],
            "passages": [{"n": 1, "citation": "c", "content": "p"}],
            "config": {"judge_model": "fake"},
        }
        for rid in sorted(ids)
    ]
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def lenient(question, claims, passages):
    return [Verdict(True, True, "ok") for _ in claims]


def strict(question, claims, passages):
    return [Verdict(c.cited == [1] and c.text.startswith("orig"), c.text.startswith("orig"), "") for c in claims]


def test_a_judge_that_approves_everything_is_exposed(tmp_path):
    summary = probe(run_file(tmp_path), tmp_path / "out.json", judge=lenient)
    assert summary["wrong_claims"] > 0 and summary["wrong_caught"] == 0
    assert summary["miscited_caught"] == 0


def test_a_judge_that_checks_claims_catches_wrong_and_miscited_but_passes_controls(tmp_path):
    out = tmp_path / "out.json"
    summary = probe(run_file(tmp_path), out, judge=strict)
    assert summary["wrong_caught"] == summary["wrong_claims"]
    assert summary["miscited_caught"] == summary["miscited"] > 0
    assert summary["controls_wrongly_flagged"] == 0
    assert json.loads(out.read_text())["cases_run"] == len(CASES)


def test_partial_results_survive_a_judge_failure(tmp_path):
    calls = []

    def dies_on_third(question, claims, passages):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("quota")
        return strict(question, claims, passages)

    out = tmp_path / "out.json"
    try:
        probe(run_file(tmp_path), out, judge=dies_on_third)
    except RuntimeError:
        pass
    saved = json.loads(out.read_text())
    assert saved["cases_run"] == 2 and saved["cases_planned"] == len(CASES)
