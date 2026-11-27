# Retrieval evaluation (M4)

Golden set: `evals/golden/retrieval_golden.jsonl`. It has 17 questions written
from the policy documents: 15 answerable, each with its expected chunk and an
evidence phrase checked by a test, plus 2 unanswerable. The expected chunks
were chosen by reading the documents, not taken from system output. All three
modes were scored on the same 15 answerable questions in one run.

Command: `python -m evals.eval_retrieval`
Raw results: `evals/results/retrieval_eval.json`

Embedding model: BAAI/bge-small-en-v1.5. Reranker: Xenova/ms-marco-MiniLM-L-6-v2.
Candidates per method: 10. RRF k: 60. Chunks: 41 from 4 documents.

## Results

| Mode | hit@1 | hit@3 | recall@3 | precision@3 | MRR | Median latency |
|---|---|---|---|---|---|---|
| dense (baseline) | 0.800 | 1.000 | 1.000 | 0.400 | 0.889 | 43 ms |
| hybrid (dense + BM25, RRF) | 0.867 | 1.000 | 1.000 | 0.400 | 0.933 | 26 ms |
| hybrid + rerank | **0.933** | 1.000 | 1.000 | 0.400 | **0.956** | 1,384 ms |

precision@3 is at its ceiling for all modes. Most questions have one relevant
chunk, so the best possible mean is 0.4.

## Per-question rank of the first relevant chunk

| id | type | dense | hybrid | hybrid + rerank |
|---|---|---|---|---|
| r05 | paraphrase | 1 | 1 | 3 |
| r12 | paraphrase | 2 | 1 | 1 |
| r13 | domain must-have (self-declared doctor) | 2 | 2 | 1 |
| r14 | paraphrase (deflection script) | 3 | 2 | 1 |
| all other 11 | | 1 | 1 | 1 |

## Decision

Ship **hybrid + rerank** (`CAREFLOW_RETRIEVAL_MODE=hybrid_rerank`).

- Hybrid beats dense on hit@1 and MRR with no latency cost.
- The reranker gives the best ranking, and it is the only mode that puts the
  zero-exception rule first for the self-declared-doctor question (r13). For a
  clinic assistant, the safety rules must come first.
- Cost: about 1.4 s extra per query on a laptop CPU. That is acceptable next
  to the LLM call that follows it. If latency matters more later, set
  `CAREFLOW_RETRIEVAL_MODE=hybrid` with no code change.

## Known gaps

- r05: the reranker drops "heart ultrasound" (echocardiogram) from rank 1 to
  rank 3. The chunk stays in the top 3, so the answer step still sees it.
- 15 scored questions over 4 documents is a small set. One question changes
  hit@1 by 0.067.