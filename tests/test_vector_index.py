"""Tests for the Qdrant index, using a tiny fake embedder.

The fake embedder counts a few keywords, so these tests need no model
download, no network and run in under a second. Qdrant runs in memory.
"""

import pytest
from qdrant_client import QdrantClient

from careflow.ingest import Chunk
from careflow.vector_index import build_index, dense_search

KEYWORDS = ("mri", "copay", "referral", "escalation")


class KeywordEmbedder:
    """Turns text into a vector of keyword counts (plus 1 to avoid zeros)."""

    def _vector(self, text):
        text = text.lower()
        return [text.count(word) + 0.01 for word in KEYWORDS]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


def _chunk(number, slug, section, text):
    return Chunk(
        chunk_id=f"{slug}#{number:03d}",
        doc_slug=slug,
        doc_title=slug.title(),
        doc_category="insurance",
        section=section,
        text=text,
        source_path=f"corpus/markdown/{slug}.md",
        loader="markdown",
    )


CHUNKS = [
    _chunk(0, "preauth", "3.2 Orthopaedics", "MRI knee needs approval. MRI MRI."),
    _chunk(1, "copay", "4.12.2 Schedule", "Specialist copay is $25. copay copay."),
    _chunk(2, "referral", "3.5 Ladder", "Referral escalation ladder. referral."),
]


@pytest.fixture
def client():
    return QdrantClient(":memory:")


def test_build_index_stores_every_chunk(client):
    count = build_index(CHUNKS, KeywordEmbedder(), client, "test")
    assert count == 3
    assert client.count("test").count == 3


def test_search_returns_best_match_with_metadata(client):
    build_index(CHUNKS, KeywordEmbedder(), client, "test")
    hits = dense_search("How much is the copay?", 2, KeywordEmbedder(), client, "test")
    best = hits[0]
    assert best["chunk_id"] == "copay#001"
    assert best["section"] == "4.12.2 Schedule"
    assert best["source_path"] == "corpus/markdown/copay.md"
    assert "score" in best


def test_rebuilding_replaces_the_old_index(client):
    build_index(CHUNKS, KeywordEmbedder(), client, "test")
    build_index(CHUNKS[:1], KeywordEmbedder(), client, "test")
    assert client.count("test").count == 1


def test_missing_index_gives_a_clear_error(client):
    with pytest.raises(RuntimeError, match="python -m careflow.vector_index"):
        dense_search("anything", 3, KeywordEmbedder(), client, "missing")


def test_empty_chunk_list_is_rejected(client):
    with pytest.raises(ValueError):
        build_index([], KeywordEmbedder(), client, "test")