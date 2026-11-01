from app.retrieve import expand_query


def test_expand_query_spells_out_abbreviations_and_keeps_the_users_words():
    assert expand_query("Does OPT end if I transfer?") == "Does OPT (optional practical training) end if I transfer?"
    assert expand_query("what does my dso do with sevis") == (
        "what does my dso (designated school official) do with sevis (Student and Exchange Visitor Information System)"
    )


def test_expand_query_leaves_other_text_alone():
    assert expand_query("How many days of unemployment?") == "How many days of unemployment?"
    assert expand_query("I want to opt out") == "I want to opt (optional practical training) out"  # known false positive
    assert expand_query("optional practical training and OPT") == "optional practical training and OPT"
    assert expand_query("OPT then OPT again") == "OPT (optional practical training) then OPT again"
    assert expand_query("pdso") == "pdso (principal designated school official)"


def test_search_expands_for_candidates_but_reranks_on_the_original_question(monkeypatch):
    import app.retrieve as retrieve

    seen = {}
    monkeypatch.setattr(retrieve, "_dense", lambda conn, q, n: seen.setdefault("dense", q) and [1])
    monkeypatch.setattr(retrieve, "_keyword", lambda conn, q, n: [1])
    monkeypatch.setattr(retrieve, "rerank_scores", lambda q, passages: seen.setdefault("rerank", q) and [0.9])

    class Conn:
        def execute(self, *_a, **_k):
            return [(1, "eCFR", "u", "c", "text")]

    hits = retrieve.search(Conn(), "Does OPT end?", top_k=1, mode="rerank")
    assert [h.id for h in hits] == [1]
    assert seen["dense"] == "Does OPT (optional practical training) end?"
    assert seen["rerank"] == "Does OPT end?"
