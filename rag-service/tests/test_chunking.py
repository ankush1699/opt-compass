from app.chunking import content_hash, pack


def test_pack_respects_limit_and_keeps_order():
    paras = ["a" * 40, "b" * 40, "c" * 40]
    assert pack(paras, max_chars=90) == ["a" * 40 + "\n" + "b" * 40, "c" * 40]


def test_pack_keeps_an_oversized_paragraph_whole():
    assert pack(["x" * 500], max_chars=100) == ["x" * 500]


def test_pack_skips_blank_paragraphs():
    assert pack(["  ", "", "real"]) == ["real"]


def test_content_hash_is_stable_and_distinguishes_text():
    assert content_hash("same") == content_hash("same")
    assert content_hash("one") != content_hash("two")
