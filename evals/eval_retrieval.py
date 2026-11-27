"""Score the retriever on the retrieval golden set.

Every mode (dense, hybrid, hybrid_rerank) is scored on the SAME golden
questions with the same metrics, so the numbers can be compared:

  hit@k        share of questions with at least one relevant chunk in the top k
  recall@k     share of each question's relevant chunks found in the top k
  precision@k  share of the top k results that are relevant
  MRR          mean of 1 / (rank of the first relevant chunk)

Unanswerable questions have no relevant chunk, so they are left out of
these retrieval metrics (they are scored later, on the final answer).

Run:
    python -m evals.eval_retrieval

Results are saved to evals/results/retrieval_eval.json.
"""

import json
import logging
import statistics
import sys
import time
from datetime import datetime, timezone

from careflow import config
from careflow.retriever import MODES, Retriever

GOLDEN_PATH = config.PROJECT_ROOT / "evals" / "golden" / "retrieval_golden.jsonl"
RESULTS_PATH = config.PROJECT_ROOT / "evals" / "results" / "retrieval_eval.json"
K_VALUES = (1, 3, 5)


def load_golden(path=GOLDEN_PATH) -> list[dict]:
    """Read the golden set (one JSON object per line)."""
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


# ---- metrics for one question ----

def hit_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    return len(set(retrieved[:k]) & set(relevant)) / k


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


# ---- running a mode ----

def score_mode(retriever: Retriever, golden: list[dict], mode: str) -> dict:
    """Run every answerable golden question through one mode and score it."""
    top_k = max(K_VALUES)
    per_question, latencies = [], []
    for case in golden:
        if not case["relevant_chunks"]:
            continue
        start = time.perf_counter()
        hits = retriever.retrieve(case["question"], top_k=top_k, mode=mode)
        latencies.append(time.perf_counter() - start)
        retrieved = [hit["chunk_id"] for hit in hits]
        relevant = case["relevant_chunks"]
        per_question.append(
            {
                "id": case["id"],
                "type": case["type"],
                "retrieved": retrieved,
                "relevant": relevant,
                "first_relevant_rank": next(
                    (r for r, c in enumerate(retrieved, 1) if c in relevant), None
                ),
                "reciprocal_rank": reciprocal_rank(retrieved, relevant),
                **{f"hit@{k}": hit_at_k(retrieved, relevant, k) for k in K_VALUES},
                **{f"recall@{k}": recall_at_k(retrieved, relevant, k) for k in K_VALUES},
                **{f"precision@{k}": precision_at_k(retrieved, relevant, k) for k in K_VALUES},
            }
        )

    def mean(key):
        return round(statistics.mean(q[key] for q in per_question), 3)

    metrics = {"mrr": mean("reciprocal_rank")}
    for k in K_VALUES:
        for name in ("hit", "recall", "precision"):
            metrics[f"{name}@{k}"] = mean(f"{name}@{k}")
    # The first query also loads models, so report the median as well.
    metrics["latency_median_ms"] = round(statistics.median(latencies) * 1000, 1)
    metrics["latency_mean_ms"] = round(statistics.mean(latencies) * 1000, 1)
    return {"metrics": metrics, "questions": per_question}


def run(modes=MODES, retriever: Retriever | None = None) -> dict:
    golden = load_golden()
    retriever = retriever or Retriever()
    retriever.retrieve("warm-up query to load the models", mode="hybrid_rerank")
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "golden_set": str(GOLDEN_PATH.relative_to(config.PROJECT_ROOT)),
        "questions_total": len(golden),
        "questions_scored": sum(1 for c in golden if c["relevant_chunks"]),
        "embedding_model": config.EMBEDDING_MODEL,
        "reranker_model": config.RERANKER_MODEL,
        "candidates_per_method": config.RETRIEVAL_CANDIDATES,
        "rrf_k": config.RRF_K,
        "modes": {mode: score_mode(retriever, golden, mode) for mode in modes},
    }
    return report


def print_table(report: dict) -> None:
    columns = ["hit@1", "hit@3", "recall@3", "precision@3", "mrr", "latency_median_ms"]
    print(f"\n{'mode':15}" + "".join(f"{c:>19}" for c in columns))
    for mode, result in report["modes"].items():
        row = result["metrics"]
        print(f"{mode:15}" + "".join(f"{row[c]:>19}" for c in columns))
    print("\nFirst relevant rank per question (- = not in top 5):")
    modes = list(report["modes"])
    print(f"{'id':6}{'type':18}" + "".join(f"{m:>15}" for m in modes))
    questions = report["modes"][modes[0]]["questions"]
    for index, question in enumerate(questions):
        ranks = [report["modes"][m]["questions"][index]["first_relevant_rank"] for m in modes]
        print(f"{question['id']:6}{question['type']:18}"
              + "".join(f"{(r if r else '-'):>15}" for r in ranks))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    selected = tuple(sys.argv[1:]) or MODES
    result = run(selected)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_table(result)
    print(f"\nSaved to {RESULTS_PATH}")