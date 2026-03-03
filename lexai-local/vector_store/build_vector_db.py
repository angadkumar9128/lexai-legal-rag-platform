#!/usr/bin/env python
"""Build a local FAISS vector store for LexAI from gold_chunks.parquet.

Usage:
    python vector_store/build_vector_db.py --if-needed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "gold_chunks.parquet"
VECTOR_DIR = ROOT_DIR / "vector_store"
INDEX_PATH = VECTOR_DIR / "faiss_index.bin"
METADATA_PATH = VECTOR_DIR / "metadata.pkl"
MANIFEST_PATH = VECTOR_DIR / "build_manifest.json"

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REQUIRED_COLUMNS = ["chunk_id", "act_name", "section_number", "chunk_text"]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _load_manifest(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _clean_gold_chunks(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    if "char_count" not in out.columns:
        out["char_count"] = pd.NA

    out["chunk_id"] = out["chunk_id"].astype(str).str.strip()
    out["act_name"] = out["act_name"].fillna("").astype(str).str.strip()
    out["section_number"] = out["section_number"].fillna("").astype(str).str.strip()
    out["chunk_text"] = out["chunk_text"].fillna("").astype(str).str.strip()
    out["char_count"] = pd.to_numeric(out["char_count"], errors="coerce")

    out = out[(out["chunk_id"] != "") & (out["chunk_text"] != "")]
    if out.empty:
        raise ValueError("No usable rows remain after dropping null/empty chunk_text or chunk_id.")

    # Deterministic order and deterministic dedupe.
    out = out.sort_values(["chunk_id", "act_name", "section_number"], kind="mergesort")
    out = out.drop_duplicates(subset=["chunk_id"], keep="first")
    out = out.reset_index(drop=True)

    return out


def _metadata_records(df: pd.DataFrame) -> List[Dict]:
    records: List[Dict] = []
    for _, row in df.iterrows():
        char_count = row.get("char_count")
        char_count_int = int(char_count) if pd.notna(char_count) else None
        records.append(
            {
                "chunk_id": str(row["chunk_id"]),
                "act_name": str(row["act_name"]),
                "section_number": str(row["section_number"]),
                "chunk_text": str(row["chunk_text"]),
                "char_count": char_count_int,
            }
        )
    return records


def _build(
    parquet_path: Path,
    model_name: str,
    batch_size: int,
    if_needed: bool,
) -> None:
    total_start = time.perf_counter()
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Input parquet not found: {parquet_path}\n"
            "Copy exported file from Databricks to lexai-local/data/gold_chunks.parquet first."
        )

    print(f"[BUILD] Source parquet: {parquet_path}")
    source_sha = _file_sha256(parquet_path)
    print(f"[BUILD] Source SHA-256: {source_sha}")

    t_load = time.perf_counter()
    df = pd.read_parquet(parquet_path)
    df = _clean_gold_chunks(df)
    load_ms = (time.perf_counter() - t_load) * 1000.0
    row_count = len(df)
    print(f"[BUILD] Rows after cleaning/dedupe: {row_count}")

    existing_manifest = _load_manifest(MANIFEST_PATH)
    if if_needed and existing_manifest:
        ready = INDEX_PATH.exists() and METADATA_PATH.exists()
        source_match = existing_manifest.get("source_sha256") == source_sha
        rows_match = int(existing_manifest.get("row_count", -1)) == row_count
        model_match = existing_manifest.get("model_name") == model_name
        if ready and source_match and rows_match and model_match:
            print("[BUILD] No changes detected; skipping rebuild due to --if-needed.")
            return

    print(f"[BUILD] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    dim = int(model.get_sentence_embedding_dimension())
    print(f"[BUILD] Embedding dimension: {dim}")

    texts = df["chunk_text"].tolist()
    t_embed = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embed_ms = (time.perf_counter() - t_embed) * 1000.0

    t_index = time.perf_counter()
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    index_ms = (time.perf_counter() - t_index) * 1000.0
    print(f"[BUILD] FAISS index size: {index.ntotal}")

    records = _metadata_records(df)

    t_save = time.perf_counter()
    faiss.write_index(index, str(INDEX_PATH))
    with METADATA_PATH.open("wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "version": 1,
        "built_at": _now_utc(),
        "source_path": str(parquet_path),
        "source_sha256": source_sha,
        "row_count": row_count,
        "model_name": model_name,
        "embedding_dimension": dim,
        "batch_size": int(batch_size),
        "index_type": "IndexFlatL2",
        "metric": "L2",
        "index_path": str(INDEX_PATH),
        "metadata_path": str(METADATA_PATH),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save_ms = (time.perf_counter() - t_save) * 1000.0

    total_ms = (time.perf_counter() - total_start) * 1000.0
    print("[BUILD] Completed successfully.")
    print(
        json.dumps(
            {
                "load_ms": round(load_ms, 2),
                "embed_ms": round(embed_ms, 2),
                "index_ms": round(index_ms, 2),
                "save_ms": round(save_ms, 2),
                "total_ms": round(total_ms, 2),
            },
            indent=2,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LexAI FAISS vector store from parquet.")
    parser.add_argument(
        "--parquet",
        type=str,
        default=str(DATA_PATH),
        help="Path to gold_chunks parquet file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--if-needed",
        action="store_true",
        help="Skip rebuild if source hash + row count + model match manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _build(
        parquet_path=Path(args.parquet),
        model_name=args.model,
        batch_size=int(args.batch_size),
        if_needed=bool(args.if_needed),
    )


if __name__ == "__main__":
    main()

