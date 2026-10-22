from app.sources import ecfr

FIXTURE = b"""<?xml version="1.0"?>
<DIV8 N="214.2" TYPE="SECTION">
<HEAD>214.2 Special requirements</HEAD>
<P>(e) Something else.</P>
<P>(f) <I>Students in colleges, universities.</I></P>
<DIV><TABLE><CAPTION><P class="title">Table 2 to Paragraph (<E T="01">f</E>)\xe2\x80\x94Paragraph Contents</P></CAPTION>
<TBODY>
<TR><TD>(1) Admission of student</TD></TR>
<TR><TD>(2) Form I-20 or successor form</TD></TR>
<TR><TD>(3) Practical training</TD></TR>
<TR><TD>(4) OPT application and approval process</TD></TR>
</TBODY></TABLE></DIV>
<P>(1) Admission of student\xe2\x80\x94(i) Eligibility. A student may be admitted.</P>
<P>(A) The student presents a Form I-20.</P>
<P>(2) Student maintenance of Form I-20 or successor form. Keep it safe.</P>
<P>(3) Practical training. Practical training may be authorized.</P>
<P>(ii) Optional practical training\xe2\x80\x94(A) General. Students may apply.</P>
<P>(4) Site visits. DHS may visit an employer.</P>
<P>(E) Periods of unemployment. No more than 90 days.</P>
<P>(4) OPT application and approval process\xe2\x80\x94(i) Student responsibilities. Ask your DSO.</P>
<P>(g) Next paragraph.</P>
</DIV8>"""


def test_top_level_numbers_follow_the_contents_table():
    rows = ecfr.parse_f_paragraphs(FIXTURE)
    by_text = {text[:14]: n for n, text in rows}
    assert by_text["(1) Admission "] == 1
    assert by_text["(A) The studen"] == 1


def test_shortened_toc_heading_still_matches_next_in_sequence():
    rows = ecfr.parse_f_paragraphs(FIXTURE)
    assert [n for n, t in rows if t.startswith("(2) Student maintenance")] == [2]


def test_numbered_subparagraph_is_not_promoted_to_top_level():
    rows = ecfr.parse_f_paragraphs(FIXTURE)
    # "(4) Site visits" is a sub-item of (f)(3), not the real (f)(4); the real one has the TOC heading.
    assert [n for n, t in rows if t.startswith("(4) Site visits")] == [3]
    assert [n for n, t in rows if t.startswith("(E) Periods")] == [3]
    assert [n for n, t in rows if t.startswith("(4) OPT application")] == [4]


def test_chunks_carry_citation_and_stop_before_next_paragraph():
    chunks = ecfr.build_chunks(FIXTURE, "2026-10-14")
    citations = {c.citation for c in chunks}
    assert "8 CFR 214.2(f)(3) Practical training" in citations
    assert all(c.source == "eCFR" for c in chunks)
    assert not any("Next paragraph" in c.content for c in chunks)
    unemployment = next(c for c in chunks if "Periods of unemployment" in c.content)
    assert unemployment.citation == "8 CFR 214.2(f)(3) Practical training"
    assert "eCFR as of 2026-10-14" in unemployment.content
