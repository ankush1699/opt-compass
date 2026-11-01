import json

from eval import agent_eval
from eval.answer_metrics import Verdict

PARTS = [
    {"must_contain": ["aggregate of more than 90 days"], "expect_any": ["90"]},
    {"must_contain": ["aggregate of more than 150 days"], "expect_any": ["150"]},
]


def side(answer, passages, calls=1, **extra):
    return {"answer": answer, "llm_calls": calls, "input_tokens": 1000, "output_tokens": 50, "latency_s": 2.0, "steps": 3,
            "passages": [{"citation": "c", "content": p} for p in passages], **extra}


def record(single, agent, rid="a01"):
    return {"id": rid, "kind": "multi", "question": "q", "parts": PARTS, "single": single, "agent": agent}


FULL = ["may not accrue an aggregate of more than 90 days", "may not accrue an aggregate of more than 150 days"]


def write(tmp_path, records):
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_part_checks_need_the_phrase_in_one_passage_and_the_fact_in_the_answer():
    assert agent_eval.part_retrieved(FULL, PARTS[0]) and agent_eval.part_retrieved(FULL, PARTS[1])
    assert not agent_eval.part_retrieved(["aggregate of more than 90 days"], PARTS[1])
    assert agent_eval.part_stated("You get 90 days [1] and 150 days [2].", PARTS[1])
    assert not agent_eval.part_stated("You get 60 days [1].", PARTS[0])
    assert not agent_eval.part_stated(None, PARTS[0])


def test_report_fails_a_deliberately_wrong_answer(tmp_path):
    right = side("90 days [1]. 150 days [2].", FULL)
    wrong = side("30 days [1]. 100 days [2].", FULL)
    report = agent_eval.report(write(tmp_path, [record(single=wrong, agent=right)]))
    assert report["agent"]["parts_stated"] == 2 and report["agent"]["questions_all_parts_stated"] == 1
    assert report["single"]["parts_stated"] == 0 and report["single"]["questions_all_parts_stated"] == 0
    assert report["single"]["parts_retrieved"] == 2  # retrieval was fine; only the answer was wrong


def test_report_counts_abstentions_and_averages_calls(tmp_path):
    abstain = side("90 days [1]. The indexed sources do not answer this.", FULL[:1], calls=1)
    agent = side("90 days [1]. 150 days [2].", FULL, calls=3)
    report = agent_eval.report(write(tmp_path, [record(abstain, agent)]))
    assert report["single"]["contains_abstain_phrase"] == 1 and report["agent"]["contains_abstain_phrase"] == 0
    assert (report["single"]["llm_calls_mean"], report["agent"]["llm_calls_mean"]) == (1, 3)
    assert report["single"]["parts_retrieved"] == 1 and report["agent"]["parts_retrieved"] == 2


def test_report_scores_refusal_probes(tmp_path):
    def refusal(rid, expected, got):
        return {"id": rid, "kind": "refusal", "question": "q", "expected_intent": expected,
                "agent": side("x", [], status="refused", intent=got)}

    path = write(tmp_path, [record(side("a", FULL), side("a", FULL)), refusal("r01", "legal_advice", "legal_advice"), refusal("r02", "in_scope", "out_of_scope")])
    probes = agent_eval.report(path)["refusal_probes"]
    assert [p["ok"] for p in probes] == [True, False]


def test_judge_step_fills_verdicts_skips_abstentions_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "fake")
    calls = []

    def fake_judge(question, claims, passages):
        calls.append(len(claims))
        return [Verdict(True, True, "ok") for _ in claims]

    monkeypatch.setattr(agent_eval, "make_llm_judge", lambda usage: fake_judge)
    abstained = side("The indexed sources do not answer this.", FULL)
    path = write(tmp_path, [record(single=abstained, agent=side("90 days [1]. 150 days [2].", FULL, calls=2))])
    agent_eval.judge_run(path)
    assert calls == [2]  # the abstention was not judged
    saved = json.loads(path.read_text())
    assert "verdicts" not in saved["single"] and len(saved["agent"]["verdicts"]) == 2
    agent_eval.judge_run(path)
    assert calls == [2]  # already judged, nothing repeated
    report = agent_eval.report(path)
    assert report["agent"]["faithfulness"] == 1.0 and report["agent"]["judged_answers"] == 1
