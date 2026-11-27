"""Tests for document loading and chunking (no LLM, no network)."""

import json

import pytest

from careflow.ingest import load_corpus

SAMPLE_MARKDOWN = """# Sample Policy

**Document ID:** TEST-001

### 1. PURPOSE

This policy is a test.

### 2. RULES

#### 2.1 Imaging

MRI needs approval under Silver.

#### 2.2 Labs

Labs never need approval.
"""


def _make_corpus(tmp_path, markdown_text=SAMPLE_MARKDOWN, write_markdown=True):
    """Build a tiny corpus folder that looks like the real one."""
    corpus = tmp_path / "corpus"
    (corpus / "markdown").mkdir(parents=True)
    (corpus / "pdf").mkdir()
    if write_markdown:
        (corpus / "markdown" / "sample.md").write_text(markdown_text, encoding="utf-8")
    index = [
        {
            "slug": "sample",
            "title": "Sample Policy",
            "section": "insurance",
            "markdown": "corpus/markdown/sample.md",
            "pdf": "corpus/pdf/sample.pdf",
        }
    ]
    (corpus / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return corpus


def test_chunks_follow_section_headings(tmp_path):
    chunks = load_corpus(_make_corpus(tmp_path), max_chars=1200)
    sections = [chunk.section for chunk in chunks]
    assert sections == ["Sample Policy", "1. PURPOSE", "2.1 Imaging", "2.2 Labs"]


def test_every_chunk_has_metadata(tmp_path):
    for chunk in load_corpus(_make_corpus(tmp_path), max_chars=1200):
        assert chunk.chunk_id.startswith("sample#")
        assert chunk.doc_slug == "sample"
        assert chunk.doc_title == "Sample Policy"
        assert chunk.doc_category == "insurance"
        assert chunk.source_path == "corpus/markdown/sample.md"
        assert chunk.loader == "markdown"
        assert chunk.text.strip()


def test_chunk_ids_are_unique(tmp_path):
    chunks = load_corpus(_make_corpus(tmp_path), max_chars=1200)
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_long_section_is_split_between_paragraphs(tmp_path):
    paragraphs = [f"Paragraph {n}. " + "word " * 30 for n in range(6)]
    text = "### 1. LONG\n\n" + "\n\n".join(paragraphs)
    chunks = load_corpus(_make_corpus(tmp_path, text), max_chars=400)
    assert len(chunks) > 1
    assert all(chunk.section == "1. LONG" for chunk in chunks)
    # No paragraph is cut in the middle: each one appears whole in one chunk.
    for paragraph in paragraphs:
        assert any(paragraph.strip() in chunk.text for chunk in chunks)


def test_missing_files_raise_a_clear_error(tmp_path):
    corpus = _make_corpus(tmp_path, write_markdown=False)
    with pytest.raises(FileNotFoundError):
        load_corpus(corpus)


def test_real_corpus_loads_with_metadata():
    chunks = load_corpus()
    assert len(chunks) >= 20
    assert {chunk.doc_slug for chunk in chunks} == {
        "referral-policy",
        "preauth-matrix",
        "copay-schedule",
        "clinical-escalation",
    }
    mri = [chunk for chunk in chunks if "ORTH-MRI" in chunk.text]
    assert mri and mri[0].section == "3.2 Orthopaedics (ORTH)"