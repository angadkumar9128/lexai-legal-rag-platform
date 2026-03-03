# LexAI Legal RAG Platform

Production-oriented legal intelligence platform for Indian law and compliance use cases, built with Databricks Lakehouse for preprocessing and a retrieval/generation stack for question answering.

## 1. Executive Summary

LexAI ingests legal PDFs, structures them into legal sections and semantic chunks, generates embeddings, and serves citation-rich legal answers.

Current recommended operating model:

- Databricks: ingestion, structuring, chunking, batch embedding generation, governed storage.
- Local or app-serving runtime: low-latency retrieval, reranking, answer generation, FastAPI/Streamlit UI.

This split gives:

- Lower online latency.
- Better control of serving stack and debugging.
- Cleaner scaling path (data pipeline and serving pipeline scale independently).

## 2. Business and Technical Goals

- Build a reliable legal knowledge pipeline over large legal corpora.
- Support natural-language legal/compliance queries.
- Return structured responses with section references and citations.
- Keep pipeline idempotent and restart-safe across Databricks serverless sessions.
- Prepare for enterprise controls (governance, lineage, auditability, reproducibility).

## 3. Scope and Non-Scope

In scope:

- Medallion data engineering pipeline (Bronze, Silver, Gold).
- Embedding generation and persistence.
- Retrieval and answer pipeline notebooks.
- API/UI wrapper for notebook-06 runtime.

Out of scope (current branch):

- Full model fine-tuning pipeline.
- Court-grade legal validation.
- Guaranteed jurisdictional legal advice.

## 4. Repository Structure

```text
lexai-legal-rag-platform/
  apps/
    fastapi_app.py
    streamlit_app.py
    lexai06_notebook_adapter.py
    notebook_06_snapshot.ipynb
    notebook_06_snapshot.json
    requirements.txt
    schemas.py
  notebooks/
    01_bronze_ingestion.ipynb
    02_silver_processing.ipynb
    03_gold_chunking.ipynb
    04_generate_embeddings.ipynb
    04_generate_embedding_Test.ipynb
    05_rag_answer_pipeline.ipynb
    05_rag_answer_pipeline_test.ipynb
    06_High-precision_QA_Legal_Reasoning_Engine.ipynb
    07_one_click_lexai_runner.ipynb
  lexai_Data_file_structure.md
  lexai-legal-rag-platform_working_docs.md
  README.md
```

## 5. Data Domains

The project is designed to support corpus folders such as:

- `constitution`
- `acts`
- `criminal_law`
- `civil_law`
- `family_law`
- `traffic_rules`
- `judgments`
- `reports`

Typical source path:

`/Volumes/workspace/legal_data/raw_documents/legal_datasets/`

## 6. End-to-End Architecture

```text
Raw PDFs
  -> Bronze: text + metadata extraction
  -> Silver: legal section structuring
  -> Gold: semantic chunking
  -> Embeddings: vectorization + artifact persistence
  -> Retrieval: lexical + dense + rerank
  -> QA: formatted legal answer with citations
  -> API/UI: FastAPI and Streamlit
```

## 7. Databricks Storage and Catalog

Base volume:

`/Volumes/workspace/legal_data/`

Key physical paths:

- Bronze files: `/Volumes/workspace/legal_data/bronze/legal_documents/`
- Silver files: `/Volumes/workspace/legal_data/silver/legal_sections/`
- Gold files: `/Volumes/workspace/legal_data/gold/legal_chunks/`
- Embedding delta artifacts: `/Volumes/workspace/legal_data/vector_db_test/legal_embeddings_delta/`
- Optional Chroma files: `/Volumes/workspace/legal_data/chroma_db/` and `/Volumes/workspace/legal_data/vector_db_test/`

Typical managed table names used in this project family:

- `workspace.default.bronze_legal_documents`
- `workspace.default.silver_legal_sections`
- `workspace.default.gold_legal_chunks`

Note:

- Use your actual catalog/schema if different.
- Delta embedding storage is the durable source of truth for recovery across serverless restarts.

## 8. Notebook Responsibilities

### `01_bronze_ingestion.ipynb`

- Parse raw PDFs.
- Extract text and metadata.
- Persist bronze records.

### `02_silver_processing.ipynb`

- Segment documents into legal sections.
- Persist section-level structure.

### `03_gold_chunking.ipynb`

- Create semantic chunks from section text.
- Preserve legal context boundaries.

### `04_generate_embedding_Test.ipynb`

- Generate embeddings from Gold chunks.
- Persist vectors and metadata to durable Delta path.
- Optionally build Chroma runtime artifacts.

### `05_rag_answer_pipeline_test.ipynb`

- Retrieval-first QA flow.
- Pull top legal context for a query.
- Basic structured legal response.

### `06_High-precision_QA_Legal_Reasoning_Engine.ipynb`

- Higher-precision retrieval logic (routing, lexical/dense blend, rerank).
- Better response structure and evidence packaging.
- Latency metrics and confidence-aware behavior.

### `07_one_click_lexai_runner.ipynb`

- Orchestrator notebook for quick setup and app bootstrapping.
- Useful for smoke testing.
- On free serverless, UI URL exposure may still be limited by platform networking.

## 9. Runtime Components

### Retrieval and QA

- Embedder: `sentence-transformers/all-MiniLM-L6-v2`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- LLM backend: notebook-06 configured backend (Databricks endpoint or local fallback path depending on runtime)

### Serving Layer

- FastAPI app: `apps/fastapi_app.py`
- Streamlit app: `apps/streamlit_app.py`
- Notebook adapter: `apps/lexai06_notebook_adapter.py`

Adapter behavior:

- Loads selected cells from notebook-06 runtime logic.
- Supports workspace notebook export fallback and snapshot fallback.
- Returns normalized response payload (answer, sections, citations, evidence, latency).

## 10. Recommended Architecture Decision

For minimum query latency, flexibility, and future scale, use:

1. Databricks for preprocessing and batch embedding generation.
2. Local or dedicated app-serving runtime for retrieval, rerank, generation, API/UI.

Do not use Databricks serverless as the only UI hosting layer for production-style app access. Keep it for pipeline compute and data governance.

## 11. Deployment Profiles

### Profile A: Databricks-native (works on current codebase)

- Use notebooks for preprocessing and QA.
- Use adapter-based API/UI from Databricks runtime.
- Best for development and validation.
- On free serverless, browser URL exposure can be unstable.

### Profile B: Hybrid (recommended production direction)

- Databricks: preprocessing and batch embeddings.
- Local or dedicated app runtime: retrieval, rerank, generation, API/UI.
- Requires local-serving engine that does not depend on Databricks `dbutils` at query time.
- This is the recommended next phase for stable UI hosting and lower latency control.

## 12. Setup Guide

### 12.1 Databricks Prerequisites

- Unity Catalog enabled workspace.
- Volume available at `/Volumes/workspace/legal_data/`.
- Access to create/read Delta tables in target catalog/schema.
- Required library installs in notebook runtime (`sentence-transformers`, `transformers`, `accelerate`, `mlflow`, `databricks-sdk`, plus app packages).

### 12.2 Local Prerequisites

- Python 3.10+
- Git
- Optional: GPU runtime for faster rerank/generation

From repo root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r apps/requirements.txt
```

Additional ML/runtime packages may be required depending on notebook-06 logic:

```bash
pip install sentence-transformers transformers accelerate mlflow databricks-sdk
```

Important:

- Current `apps/lexai06_notebook_adapter.py` executes notebook-06 logic and expects Spark-enabled runtime.
- Pure local serving without Databricks dependencies should be treated as a migration step (next phase), not a guaranteed current-state path.

## 13. Runbook

### 13.1 Data Pipeline Run Order (Databricks)

Run in order:

1. `01_bronze_ingestion.ipynb`
2. `02_silver_processing.ipynb`
3. `03_gold_chunking.ipynb`
4. `04_generate_embedding_Test.ipynb`
5. `05_rag_answer_pipeline_test.ipynb`
6. `06_High-precision_QA_Legal_Reasoning_Engine.ipynb`

Use `07_one_click_lexai_runner.ipynb` for convenience, not as a strict production control plane.

### 13.2 API Run

```bash
uvicorn apps.fastapi_app:app --host 0.0.0.0 --port 8000
```

Health:

```bash
curl http://127.0.0.1:8000/health
```

Answer:

```bash
curl -X POST http://127.0.0.1:8000/v1/legal/answer \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"What is the penalty for not wearing a helmet?\",\"style\":\"short\",\"word_limit\":120}"
```

### 13.3 Streamlit Run

```bash
streamlit run apps/streamlit_app.py
```

Optional API target:

```bash
export LEXAI_API_BASE_URL=http://127.0.0.1:8000
```

Direct-engine mode:

```bash
export LEXAI_STREAMLIT_DIRECT_ENGINE=1
```

## 14. Response Contract

Primary response fields:

- `answer`
- `sections`
- `citations[]`
- `evidence[]`
- `mode`
- `retrieval_source`
- `confidence`
- `latency_ms`
- `latency_profile`

This contract is exposed by FastAPI and used by Streamlit.

## 15. Quality Controls

- Confidence-aware fallback when evidence quality is low.
- Citation-first output style to reduce hallucination risk.
- Retrieval latency profiling (candidate fetch, lexical, dense, rerank, generation).
- Preference-driven output length/style in query handling.

## 16. Performance Guidance

- Keep lexical artifacts persisted; avoid rebuilding every session.
- Use shortlist-first retrieval, then dense and rerank on shortlist.
- Cache model loads per process.
- Keep embedding artifacts in Delta; load once into in-memory runtime index.
- Use batching in embed/rerank where possible.

## 17. Reliability and Idempotency Guidance

- Avoid ephemeral-only storage for critical artifacts.
- Persist embeddings in Delta path under volumes.
- Use deterministic IDs (`doc_id`, `section_id`, `chunk_id`) to support re-runs.
- Implement overwrite/upsert rules explicitly for repeated notebook runs.
- In serverless, expect session restarts and reinitialize runtime metadata each run.

## 18. Databricks Serverless Caveats

Known limitations on free/serverless environments:

- Driver-proxy URL may return `connection refused`.
- Cluster/session IDs change frequently.
- Previous proxy URLs become invalid after compute restart.
- Long-running app processes may be interrupted or not externally routable.

Practical rule:

- Treat notebook UI serving on free serverless as best-effort.
- Use notebook direct-engine outputs for validation.
- Use local runtime or managed serving for stable app access.

## 19. Troubleshooting

### Error: `INVALID_STATE ... cluster is Terminated`

Cause:

- URL built with stale cluster ID.

Action:

- Reattach compute.
- Rerun metadata cells.
- Use newly printed URL only from current session.

### Error: `upstream connect error ... connection refused`

Cause:

- Driver-proxy cannot reach app port in serverless runtime.

Action:

- Use direct-engine notebook mode for testing.
- Move serving UI/API to local runtime.

### Error: `No module named streamlit`

Action:

- Install with `%pip install streamlit>=1.36` in runtime.

### Error: `cannot import name TypeIs from typing_extensions`

Action:

- Upgrade `typing_extensions>=4.6.0` and restart Python runtime.

### Error: notebook path not found in adapter

Action:

- Use Databricks workspace object path (often without `.ipynb` in Repos).
- Ensure snapshot fallback file exists in `apps/`.

### ngrok DNS/connect errors

Cause:

- Runtime DNS egress blocked.

Action:

- Disable ngrok fallback.
- Use driver-proxy if available, otherwise run UI locally.

## 20. Security and Governance

- Keep source legal data in governed Unity Catalog locations.
- Restrict access to sensitive datasets by catalog/schema permissions.
- Use audit logs and lineage for regulated environments.
- Treat generated answers as assistive output, not final legal advice.

## 21. Roadmap

Near-term:

- Formal offline evaluation suite (`precision@k`, `recall@k`, citation hit rate).
- Better domain routing (`acts`/`constitution` first, case law when requested).
- Hybrid lexical+dense retrieval tuning with hard negative tests.

Mid-term:

- Decouple notebook runtime into reusable Python package.
- Dedicated service deployment (containerized API + worker + vector service).
- Policy guardrails and compliance audit traces.

Long-term:

- Fine-tuned legal generation model pipeline.
- Multi-jurisdiction and multilingual legal reasoning support.

## 22. Disclaimer

This project provides AI-generated legal information for research and assistance. It is not a substitute for professional legal advice or representation.
