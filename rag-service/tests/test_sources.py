from app.sources import federal_register, manual

FR_XML = b"""<RULE><PREAMB>
<HD SOURCE="HD1">A. Purpose</HD>
<P>""" + b"The rule explains why training plans matter. " * 8 + b"""</P>
<HD SOURCE="HD1">Authority:</HD>
<P>""" + b"Amendatory text that duplicates the eCFR. " * 8 + b"""</P>
</PREAMB></RULE>"""


def test_federal_register_skips_authority_text_and_cites_heading():
    chunks = federal_register.build_chunks(FR_XML, "https://example.test/doc")
    assert chunks, "expected the Purpose section to be indexed"
    assert all("Authority" not in c.citation for c in chunks)
    assert chunks[0].citation.startswith("81 FR 13040")
    assert "A. Purpose" in chunks[0].citation


def test_manual_pages_need_a_source_line(tmp_path):
    (tmp_path / "good.txt").write_text("Source: https://example.test/page\nTitle: A page\nBody text here.\n")
    (tmp_path / "bad.txt").write_text("no source line\nBody\n")
    chunks = manual.build_chunks(tmp_path)
    assert len(chunks) == 1
    assert chunks[0].source_url == "https://example.test/page"
    assert chunks[0].citation.startswith("A page")
