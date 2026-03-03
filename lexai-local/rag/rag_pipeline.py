"""LexAI local RAG orchestration pipeline."""

from __future__ import annotations

import time
from typing import Dict

from rag.generator import generate_answer
from rag.retriever import retrieve_chunks


def _build_context(sources: list[dict]) -> str:
    parts = []
    for i, row in enumerate(sources, start=1):
        parts.append(
            (
                f"[Source {i}]\n"
                f"Act: {row.get('act_name', '')}\n"
                f"Section: {row.get('section_number', '')}\n"
                f"Chunk ID: {row.get('chunk_id', '')}\n"
                f"Text: {row.get('chunk_text', '')}\n"
            )
        )
    return "\n".join(parts).strip()


def ask_lexai(question: str, top_k: int = 5, rerank: bool = False) -> Dict:
    """Run full local RAG for a legal query."""
    total_start = time.perf_counter()
    q = (question or "").strip()
    if not q:
        return {
            "answer": "Query cannot be empty.",
            "sources": [],
            "meta": {
                "error": "empty_query",
                "retrieve_ms": 0.0,
                "generate_ms": 0.0,
                "total_ms": 0.0,
                "rerank_used": bool(rerank),
            },
        }

    t_retrieve = time.perf_counter()
    sources = retrieve_chunks(q, top_k=int(top_k), rerank=bool(rerank))
    retrieve_ms = (time.perf_counter() - t_retrieve) * 1000.0

    context = _build_context(sources)
    t_generate = time.perf_counter()
    answer = generate_answer(context=context, question=q, style="structured")
    generate_ms = (time.perf_counter() - t_generate) * 1000.0
    total_ms = (time.perf_counter() - total_start) * 1000.0

    return {
        "answer": answer,
        "sources": sources,
        "meta": {
            "retrieve_ms": round(retrieve_ms, 2),
            "generate_ms": round(generate_ms, 2),
            "total_ms": round(total_ms, 2),
            "rerank_used": bool(rerank),
            "top_k": int(top_k),
            "source_count": len(sources),
        },
    }

