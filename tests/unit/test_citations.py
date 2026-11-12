from apex.generate.citations import check_citations, parse_citation_ids, sentence_citations


def test_parse_forms():
    assert parse_citation_ids("A [1]. B [2, 3]. C [3][4]. D [4; 5].") == [1, 2, 3, 4, 5]


def test_parse_ignores_non_numeric_brackets():
    assert parse_citation_ids("See [a] and [1a] and [ 2 ].") == [2]


def test_sentence_pairs_attach_trailing_citation():
    pairs = sentence_citations("Odysseus is from Ithaca. [2] He sails home [3].")
    assert pairs[0][1] == [2]
    assert pairs[1][1] == [3]
    assert len(pairs) == 2


def test_check_flags_invalid_and_uncited():
    res = check_citations("X is Y [1]. Z is W [7]. Q holds.", declared=[1, 2], n_evidence=3)
    assert res["valid"] == [1, 2]
    assert res["invalid"] == [7]
    assert res["declared_not_inline"] == [2]
    assert res["uncited_sentences"] == ["Z is W [7].", "Q holds."]


def test_check_tolerates_junk_declared():
    res = check_citations("A [1].", declared=["x", None, "2"], n_evidence=2)
    assert res["valid"] == [1, 2]
