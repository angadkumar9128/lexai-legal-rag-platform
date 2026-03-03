"""Streamlit UI for LexAI local RAG."""

from __future__ import annotations

import streamlit as st

from rag.generator import generator_status
from rag.rag_pipeline import ask_lexai
from rag.retriever import retriever_status


st.set_page_config(page_title="LexAI Legal RAG - Local Edition", layout="wide")

st.title("LexAI Legal RAG - Local Edition")
st.caption(
    "Fully local legal QA pipeline using Parquet + Sentence-Transformers + FAISS + local HuggingFace generation."
)

ret_status = retriever_status()
gen_status = generator_status()

with st.sidebar:
    st.subheader("Runtime Status")
    st.write(f"Retriever Ready: `{ret_status.get('ready')}`")
    st.write(f"Generator Ready: `{gen_status.get('ready')}`")
    st.write(f"Generation Model: `{gen_status.get('model', 'unknown')}`")
    st.write(f"Corpus Size: `{ret_status.get('corpus_size', 0)}`")
    top_k = st.slider("Top-K Retrieval", min_value=1, max_value=12, value=5)
    rerank = st.checkbox("Enable Reranker (slower, higher precision)", value=False)

if not ret_status.get("ready"):
    st.error(
        "Retriever is not ready. Build the vector store first:\n\n"
        "`python vector_store/build_vector_db.py --if-needed`"
    )
    st.stop()

if not gen_status.get("ready"):
    st.error(f"Generator failed to load: {gen_status.get('error', 'unknown error')}")
    st.stop()

question = st.text_area(
    "Ask a legal question",
    value="What is the penalty for not wearing a helmet?",
    height=120,
)

if st.button("Ask LexAI", type="primary"):
    with st.spinner("Retrieving legal context and generating answer..."):
        try:
            result = ask_lexai(question=question, top_k=top_k, rerank=rerank)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    st.subheader("Generated Answer")
    st.markdown(result["answer"])

    meta = result.get("meta", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Retrieve (ms)", meta.get("retrieve_ms", 0.0))
    c2.metric("Generate (ms)", meta.get("generate_ms", 0.0))
    c3.metric("Total (ms)", meta.get("total_ms", 0.0))

    st.subheader("Retrieved Sources")
    sources = result.get("sources", [])
    if not sources:
        st.info("No relevant sources were retrieved.")
    else:
        for i, src in enumerate(sources, start=1):
            title = (
                f"{i}. {src.get('act_name', 'Unknown Act')} | "
                f"Section {src.get('section_number', 'N/A')} | "
                f"chunk_id={src.get('chunk_id', 'N/A')}"
            )
            with st.expander(title):
                st.write(src.get("chunk_text", ""))
                st.caption(
                    f"rank={src.get('rank')} | score={src.get('score', 0):.5f}"
                    + (f" | rerank_score={src.get('rerank_score', 0):.5f}" if "rerank_score" in src else "")
                )

