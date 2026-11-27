"""Checks on the retrieval golden set and the metric functions."""

from careflow.ingest import load_corpus
from evals.eval_retrieval import (
    hit_at_k,
    load_golden,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_golden_set_has_enough_questions():
    golden = load_golden()
    answerable = [case for case in golden if case["relevant_chunks"]]
    assert len(answerable) >= 10
    assert any(case["type"] == "unanswerable" for case in golden)
    assert any(case["type"] == "near_duplicate" for case in golden)
    assert any(case["type"] == "exact_id" for case in golden)


def test_golden_ids_are_unique():
    ids = [case["id"] for case in load_golden()]
    assert len(ids) == len(set(ids))


def test_every_expected_chunk_exists_and_contains_its_evidence():
    """Guards against typos and against chunk numbers shifting later."""
    chunks = {chunk.chunk_id: chunk for chunk in load_corpus()}
    for case in load_golden():
        for chunk_id in case["relevant_chunks"]:
            assert chunk_id in chunks, f"{case['id']}: unknown chunk {chunk_id}"
            assert case["evidence"] in chunks[chunk_id].text, (
                f"{case['id']}: '{case['evidence']}' not found in {chunk_id}"
            )


def test_metrics_on_a_known_example():
    retrieved = ["a", "x", "b", "y", "z"]
    relevant = ["a", "b"]
    assert hit_at_k(retrieved, relevant, 1) == 1.0
    assert recall_at_k(retrieved, relevant, 1) == 0.5
    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert precision_at_k(retrieved, relevant, 3) == 2 / 3
    assert reciprocal_rank(retrieved, relevant) == 1.0
    assert reciprocal_rank(["x", "y", "a"], relevant) == 1 / 3
    assert reciprocal_rank(["x", "y"], relevant) == 0.0