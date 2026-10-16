"""Embedding model wrapper. Owner: Person 1."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.settings import EMBED_BATCH_SIZE, EMBEDDING_MODEL

# e5 models need these prefixes to tell questions from documents
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(EMBEDDING_MODEL)


def _encode(texts: list[str]) -> list[list[float]]:
    """Encode texts into normalized vectors (cosine similarity = dot product)."""
    vectors = _model().encode(
        texts, batch_size=EMBED_BATCH_SIZE, normalize_embeddings=True
    )
    return vectors.tolist()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed document chunks for indexing.

    Args:
        texts: Chunk texts.
    Returns:
        One vector per chunk.
    """
    return _encode([PASSAGE_PREFIX + t for t in texts])


def embed_query(text: str) -> list[float]:
    """Embed one user question for searching.

    Args:
        text: The question.
    Returns:
        A single vector.
    """
    return _encode([QUERY_PREFIX + text])[0]
