import json
from types import SimpleNamespace

import pytest

from app import agent
from app.generate import ABSTAIN_PHRASE
from app.retrieve import Hit


def hit(i, source="eCFR", citation="8 CFR 214.2(f)(10) Practical training"):
    return Hit(i, source, "https://example.test", citation, f"passage {i}", 0.9)


class ScriptedProvider:
    """Returns the queued replies in order and records every request, so no model is called."""

    def __init__(self, *replies, cost=0.0):
        self.replies, self.requests, self.cost = list(replies), [], cost

    def complete(self, system, user, max_tokens, json_schema=None):
        self.requests.append((system, user, json_schema))
        reply = self.replies.pop(0)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return SimpleNamespace(text=text, input_tokens=100, output_tokens=10, stop_reason="end_turn", cost_usd=self.cost)


def plan(intent="in_scope", parts=("part one", "part two")):
    return {"intent": intent, "reason": "r", "sub_questions": list(parts)}


def synth(answer="Answer [1][2].", missing=()):
    return {"answer": answer, "missing_queries": list(missing)}


class FakeSearch:
    def __init__(self, results):
        self.results, self.calls = results, []

    def __call__(self, conn, query, top_k, mode):
        self.calls.append((query, top_k, mode))
        return self.results.get(query, [])


def run(provider, search, **kwargs):
    return agent.run_agent("a question with two parts", conn=None, provider=provider, search_fn=search, **kwargs)


def test_multi_part_question_is_split_searched_per_part_and_answered_once():
    provider = ScriptedProvider(plan(), synth())
    search = FakeSearch({"part one": [hit(1), hit(2)], "part two": [hit(2), hit(3, "Federal Register", "81 FR 13040 x")]})
    result = run(provider, search)
    assert [c[0] for c in search.calls] == ["part one", "part two"] and {c[2] for c in search.calls} == {"rerank"}
    assert [h.id for h in result.sources] == [1, 2, 3]  # merged and de-duplicated, in order of first appearance
    assert result.status == "answered" and result.answer == "Answer [1][2]."
    assert (result.llm_calls, result.steps, result.input_tokens, result.cost_usd) == (2, 3, 200, 0.0)
    synthesis_prompt = provider.requests[1][1]
    assert "[3] COMMENTARY" in synthesis_prompt and "[1] REGULATION (current law)" in synthesis_prompt
    assert "a question with two parts" in synthesis_prompt


@pytest.mark.parametrize("intent", ["legal_advice", "out_of_scope"])
def test_refusal_path_stops_before_any_search(intent):
    provider = ScriptedProvider(plan(intent, parts=()))
    search = FakeSearch({})
    result = run(provider, search)
    assert result.status == "refused" and result.intent == intent
    assert search.calls == [] and result.sources == [] and result.llm_calls == 1
    assert result.answer == agent.REFUSAL[intent]
    assert result.trace[-1] == f"refuse: {intent}"


def test_legal_advice_refusal_points_to_a_dso_or_attorney():
    assert "DSO" in agent.REFUSAL["legal_advice"] and "attorney" in agent.REFUSAL["legal_advice"]


def test_missing_parts_trigger_one_follow_up_round_and_then_stop():
    provider = ScriptedProvider(plan(), synth(f"A [1]. {ABSTAIN_PHRASE}", missing=["extra query"]), synth("A [1]. B [4]."))
    search = FakeSearch({"part one": [hit(1)], "part two": [hit(2)], "extra query": [hit(4)]})
    result = run(provider, search)
    assert [c[0] for c in search.calls] == ["part one", "part two", "extra query"]
    assert result.answer == "A [1]. B [4]." and result.llm_calls == 3 and [h.id for h in result.sources] == [1, 2, 4]


def test_a_model_that_always_wants_more_searching_cannot_loop_forever():
    always_missing = synth("no", missing=["again"])
    provider = ScriptedProvider(plan(), always_missing, always_missing, always_missing, always_missing)
    result = run(provider, FakeSearch({"part one": [hit(1)]}))
    assert result.llm_calls == 1 + agent.MAX_ROUNDS and result.status == "answered"
    assert result.steps <= agent.MAX_STEPS


def test_step_limit_stops_the_graph_and_says_so():
    provider = ScriptedProvider(plan())
    search = FakeSearch({"part one": [hit(1)]})
    result = run(provider, search, max_steps=2)  # plan, research, then no step left for the answer
    assert result.status == "step_limit" and result.llm_calls == 1
    assert "ran out of steps" in result.answer and result.sources and result.trace[-1] == "step limit reached"


def test_step_limit_of_one_stops_right_after_planning():
    result = run(ScriptedProvider(plan()), FakeSearch({}), max_steps=1)
    assert result.status == "step_limit" and result.sources == []


def test_unreadable_plan_falls_back_to_the_question_as_one_part():
    provider = ScriptedProvider("not json at all", synth())
    search = FakeSearch({"a question with two parts": [hit(1)]})
    result = run(provider, search)
    assert result.intent == "in_scope" and result.sub_questions == ["a question with two parts"]
    assert "plan unreadable" in result.trace[0] and result.status == "answered"


def test_an_unknown_intent_is_not_trusted():
    provider = ScriptedProvider(plan("bomb_making", parts=()), synth())
    result = run(provider, FakeSearch({"a question with two parts": [hit(1)]}))
    assert result.intent == "in_scope" and result.status == "answered"


def test_parts_are_capped_and_blank_parts_dropped():
    parts = ["a", " ", "b", "c", "d", "e", "f"]
    search = FakeSearch({})
    run(ScriptedProvider(plan(parts=parts), synth()), search)
    assert [c[0] for c in search.calls] == ["a", "b", "c", "d"]


def test_sources_are_capped():
    many = {f"q{i}": [hit(10 * i + j) for j in range(3)] for i in range(4)}
    result = run(ScriptedProvider(plan(parts=list(many)), synth()), FakeSearch(many))
    assert len(result.sources) == agent.MAX_SOURCES


def test_total_cost_is_unknown_when_any_call_is_unpriced():
    class Unpriced(ScriptedProvider):
        def complete(self, *args, **kwargs):
            out = super().complete(*args, **kwargs)
            out.cost_usd = None if len(self.requests) == 1 else 0.0
            return out

    result = run(Unpriced(plan(), synth()), FakeSearch({"part one": [hit(1)]}))
    assert result.cost_usd is None


def test_plan_and_synthesis_requests_use_json_schemas_and_treat_the_question_as_data():
    provider = ScriptedProvider(plan(), synth())
    run(provider, FakeSearch({"part one": [hit(1)]}))
    assert provider.requests[0][2] is agent.PLAN_SCHEMA and provider.requests[1][2] is agent.SYNTH_SCHEMA
    assert "not instructions" in provider.requests[0][0]
