"""Tests for hybrid retrieval, with fake embedder and reranker (offline)."""

import pytest
from qdrant_client import QdrantClient

from careflow.retriever import Retriever, reciprocal_rank_fusion, tokenize
from careflow.vector_index import build_index
from test_vector_index import CHUNKS, KeywordEmbedder, _chunk


class PreferWordReranker:
    """Fake reranker: scores a text by how often it contains one word."""

    def __init__(self, word):
        self.word = word
        self.calls = 0

    def score(self, query, texts):
        self.calls += 1
        return [text.lower().count(self.word) for text in texts]


EXTRA = [
    # Exact code that the keyword embedder knows nothing about.
    _chunk(3, "codes", "Codes", "Procedure ENT-AUDI is an audiology test."),
]


@pytest.fixture
def retriever():
    client = QdrantClient(":memory:")
    build_index(CHUNKS + EXTRA, KeywordEmbedder(), client, "test")
    return Retriever(KeywordEmbedder(), client, PreferWordReranker("ladder"), "test")


def test_tokenize_keeps_codes_whole():
    assert tokenize("Is ORTH-MRI covered under MERIDIAN-SILVER? See 3.2.2") == [
        "is", "orth-mri", "covered", "under", "meridian-silver", "see", "3.2.2",
    ]


def test_rrf_rewards_ids_found_by_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60)
    assert fused[0][0] == "b"
    assert {chunk_id for chunk_id, _ in fused} == {"a", "b", "c", "d"}


def test_dense_mode_uses_vectors_only(retriever):
    hits = retriever.retrieve("copay amount", top_k=1, mode="dense")
    assert hits[0]["chunk_id"] == "copay#001"
    assert hits[0]["mode"] == "dense"


def test_hybrid_finds_exact_code_that_dense_misses(retriever):
    query = "Is an MRI needed before ENT-AUDI?"
    dense = [h["chunk_id"] for h in retriever.retrieve(query, 1, "dense")]
    hybrid = [h["chunk_id"] for h in retriever.retrieve(query, 2, "hybrid")]
    assert "codes#003" not in dense
    assert "codes#003" in hybrid


def test_rerank_order_is_the_order_returned(retriever):
    hits = retriever.retrieve("referral escalation", top_k=3, mode="hybrid_rerank")
    assert hits[0]["chunk_id"] == "referral#002"  # the only chunk with "ladder"
    scores = [hit["score"] for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert retriever._reranker.calls == 1


def test_results_keep_metadata(retriever):
    for hit in retriever.retrieve("mri approval", top_k=2, mode="hybrid_rerank"):
        assert {"chunk_id", "doc_slug", "section", "source_path", "text"} <= hit.keys()


def test_unknown_mode_is_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("mri", mode="magic")


def test_empty_query_returns_nothing(retriever):
    assert retriever.retrieve("   ") == []