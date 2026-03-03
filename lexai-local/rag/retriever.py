"""Local FAISS retriever for LexAI."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT_DIR / "vector_store" / "faiss_index.bin"
METADATA_PATH = ROOT_DIR / "vector_store" / "metadata.pkl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


_INDEX = None
_METADATA: List[Dict] | None = None
_EMBEDDER: SentenceTransformer | None = None
_RERANKER: CrossEncoder | None = None
_INIT_ERROR: str | None = None


def _load_resources() -> None:
    global _INDEX, _METADATA, _EMBEDDER, _INIT_ERROR
    try:
        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError(
                "Missing vector store files. Run:\n"
                "python vector_store/build_vector_db.py --if-needed"
            )

        _INDEX = faiss.read_index(str(INDEX_PATH))
        with METADATA_PATH.open("rb") as f:
            _METADATA = pickle.load(f)
        _EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME)
        _INIT_ERROR = None
    except Exception as exc:
        _INIT_ERROR = str(exc)


def _ensure_ready() -> Tuple[faiss.Index, List[Dict], SentenceTransformer]:
    global _INDEX, _METADATA, _EMBEDDER
    if _INDEX is None or _METADATA is None or _EMBEDDER is None:
        _load_resources()
    if _INIT_ERROR is not None or _INDEX is None or _METADATA is None or _EMBEDDER is None:
        raise RuntimeError(
            f"Retriever is not ready: {_INIT_ERROR}\n"
            "Build vector store first with: python vector_store/build_vector_db.py --if-needed"
        )
    return _INDEX, _METADATA, _EMBEDDER


def _ensure_reranker() -> CrossEncoder:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder(RERANK_MODEL_NAME)
    return _RERANKER


def retrieve_chunks(question: str, top_k: int = 5, rerank: bool = False) -> List[Dict]:
    """Retrieve top-k chunks for a question.

    Args:
        question: User query string.
        top_k: Number of chunks to return.
        rerank: Whether to apply cross-encoder reranking.

    Returns:
        List of chunk dictionaries including metadata + score + rank.
    """
    query = (question or "").strip()
    if not query:
        raise ValueError("question cannot be empty")
    if int(top_k) <= 0:
        raise ValueError("top_k must be > 0")

    index, metadata, embedder = _ensure_ready()
    corpus_size = len(metadata)
    if corpus_size == 0:
        return []

    top_k = min(int(top_k), corpus_size)
    search_k = min(corpus_size, max(top_k * 4, top_k))

    query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=False)
    query_vec = np.asarray(query_vec, dtype=np.float32)
    distances, indices = index.search(query_vec, search_k)

    hits: List[Dict] = []
    for rank0, idx in enumerate(indices[0].tolist(), start=1):
        if idx < 0 or idx >= corpus_size:
            continue
        dist = float(distances[0][rank0 - 1])
        md = dict(metadata[idx])
        md["distance"] = dist
        md["score"] = 1.0 / (1.0 + dist)
        md["rank"] = rank0
        hits.append(md)

    if not hits:
        return []

    if rerank:
        reranker = _ensure_reranker()
        pairs = [(query, str(h["chunk_text"])) for h in hits]
        ce_scores = reranker.predict(pairs)
        for i, ce in enumerate(ce_scores):
            hits[i]["rerank_score"] = float(ce)
        hits.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    else:
        hits.sort(key=lambda x: x["distance"])

    out = hits[:top_k]
    for i, row in enumerate(out, start=1):
        row["rank"] = i
    return out


def retriever_status() -> Dict:
    """Return light retriever diagnostics for UI."""
    ready = _INIT_ERROR is None and _INDEX is not None and _METADATA is not None and _EMBEDDER is not None
    return {
        "ready": bool(ready),
        "error": _INIT_ERROR or "",
        "index_path": str(INDEX_PATH),
        "metadata_path": str(METADATA_PATH),
        "corpus_size": len(_METADATA) if _METADATA is not None else 0,
        "embed_model": EMBED_MODEL_NAME,
        "rerank_model": RERANK_MODEL_NAME,
    }


# Warm resources at module import (load once per process).
_load_resources()

