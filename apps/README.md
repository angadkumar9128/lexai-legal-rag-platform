# LexAI 06 API + UI Integration

This folder exposes notebook `06_High-precision_QA_Legal_Reasoning_Engine.ipynb` behind:

- FastAPI (`apps/fastapi_app.py`)
- Streamlit (`apps/streamlit_app.py`)

Both surfaces return citation-rich responses.

## 1) Prerequisites

- Run inside Databricks (recommended) or any environment with:
  - Spark session access
  - Access to embedding Delta path configured in notebook 06
  - Notebook dependencies (`sentence-transformers`, `transformers`, `pyspark`, etc.)

Install app-only dependencies:

```bash
pip install -r apps/requirements.txt
```

## 2) Start API

From repo root:

```bash
uvicorn apps.fastapi_app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Answer API:

```bash
curl -X POST http://127.0.0.1:8000/v1/legal/answer \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"What is the penalty for not wearing a helmet?\",\"style\":\"short\",\"word_limit\":120}"
```

## 3) Start Streamlit UI

```bash
streamlit run apps/streamlit_app.py
```

Optional API target override:

```bash
export LEXAI_API_BASE_URL=http://127.0.0.1:8000
```

## 4) Response Contract

`POST /v1/legal/answer` returns:

- `answer`
- `sections`
- `citations[]` with parsed `act_name` and `section` when possible
- `evidence[]` (top snippets)
- `latency_ms` (current call)
- `latency_profile` (rolling p50/p95)

## 5) Notes

- The adapter executes selected cells from notebook 06 directly (`2,3,4,5,6,7,8,9`) so retrieval logic stays aligned with your working pipeline.
- Demo/eval/fine-tuning cells are not executed by the API adapter.
- If Spark or Delta paths are unavailable, API will return `503` with a clear reason.
