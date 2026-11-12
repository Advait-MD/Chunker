from apex.index.chroma_store import Hit, Node
from apex.generate.generator import pack_evidence
from apex.ingest.chunker import chunk_pages
from apex.ingest.parsers import parse_file


def test_markdown_sections(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Intro\n\nHello *world*.\n\n## Details\n\nSome `code` here.\n\n```\nx = 1\n```\n", encoding="utf-8")
    secs = parse_file(p)
    assert [s["section"] for s in secs] == ["Intro", "Details"]
    assert "Hello world." in secs[0]["text"]
    assert "x = 1" in secs[1]["text"] and secs[1]["page_number"] == 2


def test_html_strips_boilerplate(tmp_path):
    p = tmp_path / "a.html"
    p.write_text("<html><head><title>T</title><script>evil()</script></head><body><nav>menu</nav>"
                 "<p>Real content.</p><footer>foot</footer></body></html>", encoding="utf-8")
    [unit] = parse_file(p)
    assert unit["text"] == "Real content." and unit["section"] == "T"


def test_chunks_respect_pages_and_budget():
    sent = "This is a sentence with exactly ten word tokens here."
    pages = [{"page_number": 1, "section": "A", "text": " ".join([sent] * 60)},
             {"page_number": 2, "section": "B", "text": sent}]
    chunks = chunk_pages(pages, target_tokens=100, overlap_tokens=20)
    assert all(c["token_count"] <= 100 for c in chunks)
    assert {c["page_number"] for c in chunks} == {1, 2}
    assert chunks[-1]["section"] == "B" and chunks[-1]["text"] == sent
    # overlap: consecutive chunks on page 1 share their boundary sentence
    p1 = [c for c in chunks if c["page_number"] == 1]
    assert p1[1]["start_char"] < p1[0]["end_char"]


def _hit(i, score, tokens):
    return Hit(node=Node(id=f"n{i}", text=f"t{i}", level=0, doc_id="d", source_path="s", page=i,
                         token_count=tokens), score=score, rank=i)


def test_pack_evidence_orders_and_budgets():
    ev = pack_evidence([_hit(1, 0.2, 50), _hit(2, 0.9, 80), _hit(3, 0.5, 40)], token_budget=125)
    assert [it.node_id for it in ev.items] == ["n2", "n3"]
    assert [it.cite_id for it in ev.items] == [1, 2]


def test_pack_evidence_keeps_top_hit_over_budget():
    ev = pack_evidence([_hit(1, 0.9, 500)], token_budget=10)
    assert len(ev.items) == 1
