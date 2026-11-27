"""The semantic index: policy chunks stored as vectors in Qdrant.

Each chunk is turned into an embedding (a list of numbers that captures
its meaning) and saved in a local Qdrant collection together with its
metadata (document, section, source file). Qdrant runs in local mode:
the index is a folder on disk, so it survives restarts and needs no
server.

Build the index once (and again whenever the documents change):

    python -m careflow.vector_index
"""

import atexit
import logging
from functools import lru_cache
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from careflow import config
from careflow.ingest import Chunk, load_corpus

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Anything that can turn texts into vectors."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Local embedding model from fastembed (downloaded on first use)."""

    def __init__(self, model_name: str | None = None):
        from fastembed import TextEmbedding  # slow import, so only when used

        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = TextEmbedding(self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


@lru_cache(maxsize=1)
def default_embedder() -> FastEmbedEmbedder:
    """The configured embedding model, loaded once per process."""
    return FastEmbedEmbedder()


@lru_cache(maxsize=1)
def default_client() -> QdrantClient:
    """One Qdrant client for the index folder.

    Local mode allows only one client per folder at a time, so the same
    client is reused for the whole process.
    """
    config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(config.QDRANT_PATH))
    atexit.register(client.close)  # close cleanly when the program ends
    return client


def build_index(
    chunks: list[Chunk],
    embedder: Embedder | None = None,
    client: QdrantClient | None = None,
    collection: str | None = None,
) -> int:
    """(Re)create the collection and store every chunk. Returns the count."""
    embedder = embedder or default_embedder()
    client = client or default_client()
    collection = collection or config.POLICY_COLLECTION
    if not chunks:
        raise ValueError("No chunks to index.")

    vectors = embedder.embed_documents([chunk.search_text() for chunk in chunks])

    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
    )
    client.upsert(
        collection,
        points=[
            PointStruct(id=number, vector=vector, payload=chunk.to_dict())
            for number, (chunk, vector) in enumerate(zip(chunks, vectors))
        ],
    )
    logger.info("Indexed %d chunks into '%s'", len(chunks), collection)
    return len(chunks)


def dense_search(
    query: str,
    top_k: int | None = None,
    embedder: Embedder | None = None,
    client: QdrantClient | None = None,
    collection: str | None = None,
) -> list[dict]:
    """Find the chunks closest in meaning to the query.

    Returns a list of chunk dictionaries (all the metadata plus "score"),
    best match first.
    """
    embedder = embedder or default_embedder()
    client = client or default_client()
    collection = collection or config.POLICY_COLLECTION
    top_k = top_k or config.RETRIEVAL_TOP_K

    if not client.collection_exists(collection):
        raise RuntimeError(
            f"Index '{collection}' not found. Build it with: "
            "python -m careflow.vector_index"
        )
    hits = client.query_points(
        collection, query=embedder.embed_query(query), limit=top_k
    ).points
    return [{**hit.payload, "score": hit.score} for hit in hits]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = build_index(load_corpus())
    print(f"Indexed {count} chunks into {config.QDRANT_PATH}")

    question = "Does an MRI knee need pre-authorisation under the Silver plan?"
    print(f"\nTest search: {question}")
    for hit in dense_search(question, top_k=3):
        print(f"  {hit['score']:.3f}  {hit['chunk_id']:24}  {hit['section']}")