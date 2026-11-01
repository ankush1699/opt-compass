"""Tests for the answer-eval harness itself: it must catch bad answers, not just score good ones."""
import json
import re

import pytest

from app.generate import ABSTAIN_PHRASE
from eval.answer_metrics import (
    Claim,
    ScoredAnswer,
    Verdict,
    aggregate,
    cohens_kappa,
    gate,
    is_abstention,
    split_claims,
)
from eval.judge import build_prompt, parse_verdicts

PASSAGES = [
    "Students may not accrue an aggregate of more than 90 days of unemployment during any post-completion OPT period.",
    "Students granted a 24-month OPT extension may not accrue an aggregate of more than 150 days of unemployment.",
    "A student may not begin OPT employment prior to the approved start date on the EAD.",
]


def stub_judge(claims: list[Claim], passages: list[str]) -> list[Verdict]:
    """Deterministic stand-in for the LLM judge: a claim is supported when its numbers and most of its
    words appear in a passage. Good enough to prove the harness reacts to wrong answers."""
    def words(text):
        return set(re.findall(r"[a-z0-9-]+", text.lower()))

    def supports(claim_text, passage_text):
        claim_words, passage_words = words(claim_text), words(passage_text)
        numbers = {w for w in claim_words if w.isdigit()}
        return numbers <= passage_words and len(claim_words & passage_words) / len(claim_words) >= 0.5

    verdicts = []
    for claim in claims:
        any_ok = any(supports(claim.text, p) for p in passages)
        cited = [passages[n - 1] for n in claim.cited if 1 <= n <= len(passages)]
        verdicts.append(Verdict(any(supports(claim.text, p) for p in cited), any_ok))
    return verdicts


def score(kind: str, text: str) -> ScoredAnswer:
    if is_abstention(text):
        return ScoredAnswer(kind, True, n_passages=len(PASSAGES))
    claims = split_claims(text)
    return ScoredAnswer(kind, False, claims, stub_judge(claims, PASSAGES), len(PASSAGES))


GOOD = "You may not accrue an aggregate of more than 90 days of unemployment during post-completion OPT [1]."
FABRICATED = "You may accrue an aggregate of up to 365 days of unemployment during post-completion OPT [1]."
MISCITED = "You may not accrue an aggregate of more than 90 days of unemployment during post-completion OPT [3]."


def test_split_claims_reads_markers_and_sentences():
    claims = split_claims("The limit is 90 days [1]. A STEM extension allows 150 days. [2][3]\n- Third point [2, 3].")
    assert [c.cited for c in claims] == [[1], [2, 3], [2, 3]]
    assert claims[0].text == "The limit is 90 days ."
    assert len(claims) == 3


def test_abbreviations_do_not_split_a_sentence():
    claims = split_claims("An employer may not replace a U.S. worker with the student [1].")
    assert len(claims) == 1


def test_abstention_is_detected_case_insensitively():
    assert is_abstention(ABSTAIN_PHRASE)
    assert is_abstention(ABSTAIN_PHRASE.upper())
    assert not is_abstention("The limit is 90 days [1].")
    assert not is_abstention(None)


def test_correct_answer_passes_the_gate():
    metrics = aggregate([score("answerable", GOOD), score("unanswerable", ABSTAIN_PHRASE)])
    assert metrics["faithfulness"] == 1.0
    assert metrics["citation_precision"] == 1.0
    assert gate(metrics) == []


def test_fabricated_answer_fails_the_gate():
    metrics = aggregate([score("answerable", FABRICATED), score("unanswerable", ABSTAIN_PHRASE)])
    assert metrics["faithfulness"] == 0.0
    assert any(f.startswith("faithfulness") for f in gate(metrics))


def test_wrong_citation_lowers_citation_precision_but_not_faithfulness():
    metrics = aggregate([score("answerable", MISCITED), score("unanswerable", ABSTAIN_PHRASE)])
    assert metrics["faithfulness"] == 1.0
    assert metrics["citation_precision"] == 0.0
    assert any(f.startswith("citation_precision") for f in gate(metrics))


def test_out_of_range_citation_is_counted_not_ignored():
    metrics = aggregate([score("answerable", GOOD.replace("[1]", "[9]")), score("unanswerable", ABSTAIN_PHRASE)])
    assert metrics["invalid_citation_rate"] == 1.0
    assert metrics["citation_coverage"] == 0.0


def test_guessing_on_an_unanswerable_question_fails_the_gate():
    metrics = aggregate([score("answerable", GOOD), score("unanswerable", "Premium processing costs $2,805 [1].")])
    assert metrics["abstain_rate_on_unanswerable"] == 0.0
    assert any(f.startswith("abstain_rate_on_unanswerable") for f in gate(metrics))


def test_refusing_every_answerable_question_fails_the_gate():
    metrics = aggregate([score("answerable", ABSTAIN_PHRASE), score("unanswerable", ABSTAIN_PHRASE)])
    assert metrics["false_abstain_rate"] == 1.0
    failures = gate(metrics)
    assert any(f.startswith("false_abstain_rate") for f in failures)
    assert any(f.startswith("faithfulness") for f in failures)  # no answered claims means no evidence


def test_gate_fails_closed_when_there_is_no_data():
    assert gate(aggregate([])) != []


def test_judge_that_skips_a_claim_cannot_make_it_pass():
    verdicts = parse_verdicts(json.dumps({"claims": [
        {"index": 1, "supported_by_any": True, "supported_by_cited": True, "reason": "ok"},
    ]}), n_claims=2)
    assert verdicts[0].supported_by_any and not verdicts[1].supported_by_any


def test_unparseable_judge_output_marks_everything_unsupported():
    verdicts = parse_verdicts("not json at all", n_claims=3)
    assert len(verdicts) == 3 and not any(v.supported_by_any for v in verdicts)


def test_judge_prompt_lists_passages_claims_and_their_citations():
    prompt = build_prompt("How long?", [Claim("It is 90 days", [1]), Claim("Uncited claim", [])], PASSAGES[:2])
    assert "[1]\nStudents may not accrue" in prompt
    assert "1. It is 90 days  (cites: 1)" in prompt
    assert "2. Uncited claim  (cites: nothing)" in prompt


def test_cohens_kappa_known_values():
    assert cohens_kappa([True, True, False, False], [True, True, False, False]) == 1.0
    assert cohens_kappa([True, False, True, False], [False, True, False, True]) == -1.0
    assert cohens_kappa([True, True, True], [True, True, True]) is None  # undefined when both always agree on one label
    assert cohens_kappa([], []) is None


def test_report_refuses_a_run_that_was_generated_without_the_judge(tmp_path):
    from eval.answer_eval import report

    record = {"id": "q001", "kind": "answerable", "claims": [{"text": "x", "cited": [1]}], "verdicts": []}
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(SystemExit):
        report(path)
