"""Parse PDF / Markdown / HTML / TXT into page-like units with page or section metadata.

PDF parsing reuses the existing, tested `modules.document_loader.read_pdf_pages`
(PyPDF2 + artefact cleaning) rather than introducing pymupdf (spec deviation, see docs/M1_NOTES.md).
"""

from __future__ import annotations

import json

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Dict, List


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    paras, cur = [], []
    for ln in lines:
        if ln:
            cur.append(ln)
        elif cur:
            paras.append(" ".join(cur))
            cur = []
    if cur:
        paras.append(" ".join(cur))
    return "\n\n".join(paras)


def parse_pdf(path: Path) -> List[Dict]:
    from modules.document_loader import read_pdf_pages
    return [{"page_number": p["page_number"], "section": None, "text": p["text"]} for p in read_pdf_pages(str(path))]


def parse_markdown(path: Path) -> List[Dict]:
    """One unit per heading section; `page_number` is the 1-based section index."""
    from markdown_it import MarkdownIt

    tokens = MarkdownIt("commonmark").parse(path.read_text(encoding="utf-8", errors="replace"))
    sections: List[Dict] = []
    heading, buf, in_heading = None, [], False

    def flush():
        text = _normalise("\n\n".join(buf))
        if text:
            sections.append({"page_number": len(sections) + 1, "section": heading, "text": text})

    for tok in tokens:
        if tok.type == "heading_open":
            flush()
            buf, in_heading = [], True
        elif tok.type == "heading_close":
            in_heading = False
        elif tok.type == "inline":
            content = "".join(c.content for c in (tok.children or []) if c.type in ("text", "code_inline", "softbreak"))
            content = content or tok.content
            if in_heading:
                heading = content.strip()
            else:
                buf.append(content)
        elif tok.type in ("fence", "code_block"):
            buf.append(tok.content)
    flush()
    return sections


def parse_html(path: Path) -> List[Dict]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    for tag in soup(["head", "script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    text = _normalise(soup.get_text("\n\n"))
    return [{"page_number": 1, "section": title, "text": text}] if text else []


def parse_txt(path: Path) -> List[Dict]:
    text = _normalise(path.read_text(encoding="utf-8", errors="replace"))
    return [{"page_number": 1, "section": None, "text": text}] if text else []


def parse_jsonl_doc(path: Path) -> List[Dict]:
    """Pre-segmented document: one JSON object per line, {"section": str|null, "text": str}.

    Lossless input for benchmark corpora (no Markdown re-interpretation of the text). Each line becomes
    one unit, so chunks never cross a section/paragraph boundary. Malformed lines raise.
    """
    units: List[Dict] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{n}: invalid JSON: {e}") from e
        if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
            raise ValueError(f"{path}:{n}: each line must be an object with a string 'text'")
        text = _normalise(obj["text"])
        if text:
            units.append({"page_number": len(units) + 1, "section": obj.get("section"), "text": text})
    return units


PARSERS = {".pdf": parse_pdf, ".md": parse_markdown, ".markdown": parse_markdown,
           ".html": parse_html, ".htm": parse_html, ".txt": parse_txt, ".jsonl": parse_jsonl_doc}


def parse_file(path: str | Path) -> List[Dict]:
    p = Path(path)
    parser = PARSERS.get(p.suffix.lower())
    if parser is None:
        raise ValueError(f"Unsupported file type: {p.suffix} ({p})")
    return parser(p)
