"""Hybrid retrieval over the policy index.

Three modes, chosen by CAREFLOW_RETRIEVAL_MODE (or per call):

  dense          vector search only (the baseline)
  hybrid         vector search + BM25 keyword search, merged with
                 Reciprocal Rank Fusion (RRF)
  hybrid_rerank  hybrid, then a cross-encoder reranker reads the query
                 and each candidate together and reorders them

BM25 matters for exact identifiers such as "ORTH-MRI" or
"MERIDIAN-SILVER", which vector search can blur together. The reranker
matters for picking the best of several similar chunks. The reranker's
order is the order returned, so it is the order the LLM will see.
"""

import logging
import re
from functools import lru_cache
from typing import Protocol

from rank_bm25 import BM25Okapi

from careflow import config
from careflow.vector_index import default_client, default_embedder, dense_search

logger = logging.getLogger(__name__)

MODES = ("dense", "hybrid", "hybrid_rerank")

# Words, numbers and codes. "ORTH-MRI" stays one token, "3.2.2" too.
TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lower-case tokens for BM25, keeping codes like orth-mri whole."""
    return TOKEN.findall(text.lower())


class Reranker(Protocol):
    """Anything that scores (query, text) pairs; higher is more relevant."""

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class FastEmbedReranker:
    """Local cross-encoder reranker from fastembed (downloaded on first use)."""

    def __init__(self, model_name: str | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name or config.RERANKER_MODEL
        self._model = TextCrossEncoder(self.model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(value) for value in self._model.rerank(query, texts)]


@lru_cache(maxsize=1)
def default_reranker() -> FastEmbedReranker:
    """The configured reranker, loaded once per process."""
    return FastEmbedReranker()


def _search_text(chunk: dict) -> str:
    return f"{chunk['doc_title']} | {chunk['section']}\n{chunk['text']}"


def reciprocal_rank_fusion(rankings: list[list[str]], k: int) -> list[tuple[str, float]]:
    """Merge several ranked lists of ids into one.

    Each list gives an id 1 / (k + rank) points; ids found high up in
    several lists win. Returns [(id, fused_score), ...], best first.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class Retriever:
    """Searches the policy index in one of the three modes."""

    def __init__(self, embedder=None, client=None, reranker=None, collection=None):
        self.embedder = embedder
        self.client = client
        self._reranker = reranker
        self.collection = collection or config.POLICY_COLLECTION
        self._chunks: list[dict] | None = None
        self._bm25: BM25Okapi | None = None

    # -- lazily created parts, so tests can pass fakes --

    def _get_embedder(self):
        if self.embedder is None:
            self.embedder = default_embedder()
        return self.embedder

    def _get_client(self):
        if self.client is None:
            self.client = default_client()
        return self.client

    def _get_reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = default_reranker()
        return self._reranker

    def _load_chunks(self) -> list[dict]:
        """Read every chunk from the index, so BM25 and dense see the same set."""
        if self._chunks is None:
            client = self._get_client()
            if not client.collection_exists(self.collection):
                raise RuntimeError(
                    f"Index '{self.collection}' not found. Build it with: "
                    "python -m careflow.vector_index"
                )
            chunks, offset = [], None
            while True:
                points, offset = client.scroll(
                    self.collection, limit=256, offset=offset, with_payload=True
                )
                chunks.extend(point.payload for point in points)
                if offset is None:
                    break
            chunks.sort(key=lambda chunk: chunk["chunk_id"])
            self._chunks = chunks
            self._bm25 = BM25Okapi([tokenize(_search_text(c)) for c in chunks])
        return self._chunks

    # -- the three search methods --

    def _dense(self, query: str, limit: int) -> list[dict]:
        return dense_search(
            query, limit, self._get_embedder(), self._get_client(), self.collection
        )

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        chunks = self._load_chunks()
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        return [
            {**chunks[i], "score": float(scores[i])}
            for i in ranked[:limit]
            if scores[i] > 0
        ]

    def retrieve(self, query: str, top_k: int | None = None, mode: str | None = None) -> list[dict]:
        """Return the top_k most relevant chunks for the query, best first.

        Each result is the chunk's metadata plus "score" and "mode".
        """
        top_k = top_k or config.RETRIEVAL_TOP_K
        mode = mode or config.RETRIEVAL_MODE
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        if not query.strip():
            return []

        if mode == "dense":
            return [{**hit, "mode": mode} for hit in self._dense(query, top_k)]

        candidates = config.RETRIEVAL_CANDIDATES
        dense_hits = self._dense(query, candidates)
        keyword_hits = self._bm25_search(query, candidates)
        by_id = {hit["chunk_id"]: hit for hit in keyword_hits + dense_hits}
        fused = reciprocal_rank_fusion(
            [[h["chunk_id"] for h in dense_hits], [h["chunk_id"] for h in keyword_hits]],
            k=config.RRF_K,
        )

        if mode == "hybrid":
            return [
                {**by_id[chunk_id], "score": score, "mode": mode}
                for chunk_id, score in fused[:top_k]
            ]

        # hybrid_rerank: the reranker reads every fused candidate and its
        # order is the one returned (and later shown to the LLM).
        pool = [by_id[chunk_id] for chunk_id, _ in fused]
        scores = self._get_reranker().score(query, [_search_text(c) for c in pool])
        reranked = sorted(zip(pool, scores), key=lambda pair: pair[1], reverse=True)
        return [
            {**chunk, "score": score, "mode": mode} for chunk, score in reranked[:top_k]
        ]


@lru_cache(maxsize=1)
def default_retriever() -> Retriever:
    """One shared retriever per process."""
    return Retriever()


def retrieve(query: str, top_k: int | None = None, mode: str | None = None) -> list[dict]:
    """Search the policy documents (shortcut for the shared retriever)."""
    return default_retriever().retrieve(query, top_k, mode)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    questions = [
        "Does an MRI knee need pre-authorisation under the Silver plan?",
        "What is the telehealth co-pay for MERIDIAN-BRONZE?",
        "What should staff say when a patient asks for medication advice?",
    ]
    for question in questions:
        print(f"\n{question}")
        for mode in MODES:
            hits = retrieve(question, top_k=3, mode=mode)
            print(f"  {mode:14} " + ", ".join(hit["chunk_id"] for hit in hits))