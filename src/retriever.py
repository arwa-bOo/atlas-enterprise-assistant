"""Domain-filtered retrieval. Owner: Person 1."""

from collections import Counter
from functools import lru_cache

import chromadb

from src.embeddings import embed_query
from src.schemas import Chunk
from src.settings import (
    CANDIDATE_MULTIPLIER,
    COLLECTION_NAME,
    INDEX_DIR,
    MAX_CHUNKS_PER_FILE,
    TOP_K,
)


@lru_cache(maxsize=1)
def _collection() -> chromadb.Collection:
    """Open the persisted collection, with a clear error if it is missing."""
    if not INDEX_DIR.exists():
        raise FileNotFoundError("Index not found. Run: python -m src.ingest")
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    return client.get_collection(COLLECTION_NAME)


def retrieve(query: str, domain: str, k: int = TOP_K) -> list[Chunk]:
    """Return the top-k chunks within the domain, spread across different files.

    Args:
        query: User question (Arabic or English).
        domain: "hr", "it" or "finance".
        k: Number of chunks to return.
    Returns:
        Chunks sorted by similarity, at most MAX_CHUNKS_PER_FILE per file.
    """
    result = _collection().query(
        query_embeddings=[embed_query(query)],
        n_results=k * CANDIDATE_MULTIPLIER,
        where={"domain": domain},
    )
    candidates = [
        Chunk(chunk_id=cid, text=doc, score=1 - dist, **meta)
        for cid, doc, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]
    kept: list[Chunk] = []
    per_file: Counter[str] = Counter()
    for chunk in candidates:  # already sorted by similarity
        if per_file[chunk.source_filename] < MAX_CHUNKS_PER_FILE:
            kept.append(chunk)
            per_file[chunk.source_filename] += 1
        if len(kept) == k:
            break
    return kept


def retrieve_multi(queries: list[str], domain: str, k: int = TOP_K) -> list[Chunk]:
    """Search with several phrasings (e.g. the question and its translation).

    Results are interleaved so every phrasing gets a fair share of the k slots.

    Args:
        queries: Different versions of the same question (e.g. English + Arabic).
        domain: "hr", "it" or "finance".
        k: Total number of chunks to return.
    Returns:
        Unique chunks (by chunk_id), best of each query first.
    """
    per_query = [retrieve(q, domain, k) for q in queries]
    merged: list[Chunk] = []
    seen: set[str] = set()
    for rank in range(k):
        for results in per_query:
            if rank < len(results) and results[rank].chunk_id not in seen:
                seen.add(results[rank].chunk_id)
                merged.append(results[rank])
    return merged[:k]
