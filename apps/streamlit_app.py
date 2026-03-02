import os
from typing import Dict

import requests
import streamlit as st

from apps.lexai06_notebook_adapter import get_engine

API_BASE_URL = os.environ.get("LEXAI_API_BASE_URL", "http://127.0.0.1:8000")
ANSWER_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/v1/legal/answer"
LATENCY_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/v1/legal/latency"
HEALTH_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/health"
USE_DIRECT_ENGINE = os.environ.get("LEXAI_STREAMLIT_DIRECT_ENGINE", "0").strip() == "1"
_ENGINE = None


def _normalize_direct_payload(query: str, payload: Dict) -> Dict:
    citations = [{"citation_text": str(x)} for x in payload.get("citations", [])]
    return {
        "query": query,
        "answer": str(payload.get("answer", "")).strip(),
        "sections": [str(x).strip() for x in payload.get("sections", []) if str(x).strip()],
        "citations": citations,
        "mode": str(payload.get("mode", "unknown")),
        "retrieval_source": str(payload.get("source", "unknown")),
        "confidence": str(payload.get("confidence", "na")),
        "preferences": payload.get("preferences", {}) or {},
        "latency_ms": payload.get("latency_ms", {}) or {},
        "latency_profile": payload.get("latency_profile", {}) or {},
        "evidence": payload.get("evidence", []) or [],
    }


def _direct_answer(query: str, style: str, word_limit: int) -> Dict:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = get_engine()
    q = query.strip()
    if style and style != "none":
        q += f" in {style}"
    if word_limit and int(word_limit) > 0:
        q += f" within {int(word_limit)} words"
    payload = _ENGINE.answer_query(q)
    return _normalize_direct_payload(query=query.strip(), payload=payload)


def call_api(query: str, style: str, word_limit: int) -> Dict:
    if USE_DIRECT_ENGINE:
        return _direct_answer(query=query, style=style, word_limit=word_limit)

    payload = {
        "query": query,
        "style": style if style != "none" else None,
        "word_limit": word_limit if word_limit > 0 else None,
    }
    resp = requests.post(ANSWER_ENDPOINT, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()


def fetch_json(url: str) -> Dict:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


st.set_page_config(page_title="LexAI Legal QA", layout="wide")
st.title("LexAI Legal QA Engine (Notebook 06)")
st.caption("Citation-rich legal responses via FastAPI adapter")

with st.sidebar:
    st.subheader("Configuration")
    st.write(f"`API_BASE_URL`: `{API_BASE_URL}`")
    st.write(f"`DIRECT_ENGINE_MODE`: `{USE_DIRECT_ENGINE}`")
    style = st.selectbox("Response Style", options=["none", "very short", "short", "normal", "detailed"], index=2)
    word_limit = st.number_input("Word Limit (optional)", min_value=0, max_value=400, value=120, step=10)
    show_evidence = st.checkbox("Show top evidence", value=True)
    show_latency = st.checkbox("Show latency metrics", value=True)

    if st.button("Check API Health"):
        if USE_DIRECT_ENGINE:
            health = {"ok": True, "status": "direct_engine_mode"}
        else:
            health = fetch_json(HEALTH_ENDPOINT)
        st.json(health)

query = st.text_area(
    "Ask your legal question",
    value="What is the penalty for not wearing a helmet?",
    height=110,
)

cols = st.columns([1, 4])
with cols[0]:
    run = st.button("Get Answer", type="primary")
with cols[1]:
    if st.button("Show Latency Profile"):
        if USE_DIRECT_ENGINE:
            try:
                if _ENGINE is None:
                    _ENGINE = get_engine()
                lp = {"ok": True, "latency_profile": _ENGINE.latency_profile()}
            except Exception as exc:
                lp = {"ok": False, "error": str(exc)}
        else:
            lp = fetch_json(LATENCY_ENDPOINT)
        st.json(lp)

if run:
    if not query.strip():
        st.error("Query cannot be empty.")
    else:
        try:
            data = call_api(query=query.strip(), style=style, word_limit=int(word_limit))
        except requests.HTTPError as http_err:
            detail = ""
            try:
                detail = http_err.response.json()
            except Exception:
                detail = str(http_err)
            st.error(f"API HTTP error: {detail}")
            st.stop()
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

        st.subheader("Legal Explanation")
        st.write(data.get("answer", ""))

        sections = data.get("sections", [])
        st.subheader("Relevant Sections")
        if sections:
            st.write(", ".join(sections))
        else:
            st.write("No sections returned.")

        st.subheader("Citations")
        citations = data.get("citations", [])
        if citations:
            for c in citations:
                txt = c.get("citation_text") if isinstance(c, dict) else str(c)
                st.markdown(f"- {txt}")
        else:
            st.write("No citations returned.")

        if show_evidence:
            st.subheader("Top Evidence")
            for idx, ev in enumerate(data.get("evidence", [])[:5], start=1):
                title = f"{idx}. {ev.get('act', 'Unknown Act')} | Section {ev.get('section', 'NA')} | score={ev.get('score', 'NA')}"
                with st.expander(title):
                    st.write(ev.get("snippet", ""))
                    st.caption(f"source_group: {ev.get('source_group', 'na')}")

        if show_latency:
            st.subheader("Latency (Current)")
            st.json(data.get("latency_ms", {}))
            st.subheader("Latency (p50/p95)")
            st.json(data.get("latency_profile", {}))

        st.caption("AI-generated legal information. Not a substitute for professional legal advice.")
