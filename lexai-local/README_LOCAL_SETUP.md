# LexAI Local Setup Guide

This setup runs LexAI fully local for retrieval + generation + UI.

## 1) System Requirements

- OS: Windows, Linux, or macOS
- Python: 3.10+ (recommended 3.10/3.11)
- RAM: 16 GB recommended
- Disk: 10+ GB free (model caches + index artifacts)
- Internet: required first time for model download

## 2) Project Structure

```text
lexai-local/
  data/
    gold_chunks.parquet
  vector_store/
    build_vector_db.py
    faiss_index.bin           # generated
    metadata.pkl              # generated
    build_manifest.json       # generated
  rag/
    retriever.py
    generator.py
    rag_pipeline.py
  app.py
  requirements.txt
```

## 3) Put Gold Export Locally

Export `workspace.default.gold_legal_chunks` from Databricks, then copy:

`gold_chunks.parquet` -> `lexai-local/data/gold_chunks.parquet`

Required columns:

- `chunk_id`
- `act_name`
- `section_number`
- `chunk_text`
- `char_count`

## 4) Create Virtual Environment

### Windows (PowerShell)

```powershell
cd lexai-local
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/macOS

```bash
cd lexai-local
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5) Build Local Vector Store

```bash
python vector_store/build_vector_db.py --if-needed
```

This command:

1. Validates required columns in parquet.
2. Cleans rows with null/empty text.
3. Creates embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
4. Builds FAISS `IndexFlatL2`.
5. Writes:
- `vector_store/faiss_index.bin`
- `vector_store/metadata.pkl`
- `vector_store/build_manifest.json`

## 6) Run Streamlit App

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit (usually `http://localhost:8501`).

## 7) One-Command Flow

### Windows PowerShell

```powershell
python vector_store/build_vector_db.py --if-needed; if ($LASTEXITCODE -eq 0) { streamlit run app.py }
```

### Linux/macOS

```bash
python vector_store/build_vector_db.py --if-needed && streamlit run app.py
```

## 8) Optional Model Switch

Default generator model (faster on CPU):

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

Optional higher-quality/slower model:

```bash
export LEXAI_GEN_MODEL=microsoft/phi-2
```

Windows PowerShell:

```powershell
$env:LEXAI_GEN_MODEL="microsoft/phi-2"
```

Then restart Streamlit.

## 9) Performance Notes

- Cold start (first run): slower due to model downloads.
- Warm start: faster after models and index are cached.
- FAISS retrieval should be fast on warm process.
- Reranker increases precision but adds latency.
- To reduce latency:
  - Keep reranker off unless needed.
  - Use lower `top_k` (for example 4–6).
  - Keep default fast generation model.

## 10) Troubleshooting

### A) `Input parquet not found`

Copy file to:

`lexai-local/data/gold_chunks.parquet`

### B) `Missing required columns`

Check parquet schema includes:

- `chunk_id`, `act_name`, `section_number`, `chunk_text`

### C) `Retriever is not ready`

Run:

```bash
python vector_store/build_vector_db.py --if-needed
```

### D) Model load errors or very slow first run

- Ensure internet access for first download.
- Retry after clearing partial HF cache if needed.
- Switch to default fast model if `phi-2` is too heavy.

### E) Slow CPU generation

- Keep reranker disabled.
- Reduce `top_k`.
- Set lighter generation settings via env vars:
  - `LEXAI_MAX_NEW_TOKENS`
  - `LEXAI_TEMPERATURE`
  - `LEXAI_TOP_P`

## 11) Databricks Deterministic Export (Reference Cell)

Use this Databricks code to export deterministic parquet for local use:

```python
from pyspark.sql import functions as F, Window
import json
import hashlib
from datetime import datetime, timezone

SOURCE_TABLE = "workspace.default.gold_legal_chunks"
TARGET_DIR = "/Volumes/workspace/legal_data/exports/gold_chunks/latest"
TARGET_FILE = f"{TARGET_DIR}/gold_chunks.parquet"
META_FILE = f"{TARGET_DIR}/export_metadata.json"

required_cols = ["chunk_id", "act_name", "section_number", "chunk_text", "char_count"]
df = spark.table(SOURCE_TABLE).select(*required_cols)

df = (
    df.withColumn("chunk_id", F.trim(F.col("chunk_id").cast("string")))
      .withColumn("act_name", F.trim(F.col("act_name").cast("string")))
      .withColumn("section_number", F.trim(F.col("section_number").cast("string")))
      .withColumn("chunk_text", F.trim(F.col("chunk_text").cast("string")))
      .withColumn("char_count", F.col("char_count").cast("int"))
      .filter(F.col("chunk_id").isNotNull() & (F.col("chunk_id") != ""))
      .filter(F.col("chunk_text").isNotNull() & (F.col("chunk_text") != ""))
)

w = Window.partitionBy("chunk_id").orderBy(
    F.col("act_name").asc_nulls_last(),
    F.col("section_number").asc_nulls_last(),
    F.col("chunk_text").asc_nulls_last(),
)
df = df.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") == 1).drop("rn")
df = df.orderBy(F.col("chunk_id").asc())

pdf = df.toPandas()
pdf.to_parquet(TARGET_FILE, index=False)

schema_text = "|".join(f"{c}:{str(t)}" for c, t in zip(pdf.columns, pdf.dtypes))
schema_hash = hashlib.sha256(schema_text.encode("utf-8")).hexdigest()

meta = {
    "source_table": SOURCE_TABLE,
    "target_file": TARGET_FILE,
    "row_count": int(len(pdf)),
    "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    "schema_hash": schema_hash,
}

dbutils.fs.put(META_FILE, json.dumps(meta, indent=2), overwrite=True)
print(json.dumps(meta, indent=2))
```

