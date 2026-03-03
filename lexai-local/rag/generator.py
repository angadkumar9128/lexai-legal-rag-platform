"""Local text generation module for LexAI."""

from __future__ import annotations

import os
from typing import Dict, Optional

from transformers import AutoTokenizer, pipeline


FAST_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GEN_MODEL_NAME = os.environ.get("LEXAI_GEN_MODEL", FAST_MODEL).strip() or FAST_MODEL
MAX_NEW_TOKENS = int(os.environ.get("LEXAI_MAX_NEW_TOKENS", "180"))
TEMPERATURE = float(os.environ.get("LEXAI_TEMPERATURE", "0.2"))
TOP_P = float(os.environ.get("LEXAI_TOP_P", "0.9"))

_TEXT_GEN = None
_TOKENIZER = None
_GEN_INIT_ERROR: Optional[str] = None


def _load_generator():
    global _TEXT_GEN, _TOKENIZER, _GEN_INIT_ERROR
    try:
        _TOKENIZER = AutoTokenizer.from_pretrained(
            GEN_MODEL_NAME,
            trust_remote_code=True,
        )
        _TEXT_GEN = pipeline(
            "text-generation",
            model=GEN_MODEL_NAME,
            tokenizer=_TOKENIZER,
            device=-1,  # CPU-first target
            trust_remote_code=True,
        )
        _GEN_INIT_ERROR = None
    except Exception as exc:
        _GEN_INIT_ERROR = str(exc)
        _TEXT_GEN = None
        _TOKENIZER = None


def _ensure_generator():
    if _TEXT_GEN is None:
        _load_generator()
    if _TEXT_GEN is None:
        raise RuntimeError(
            f"Generator model failed to load: {_GEN_INIT_ERROR}\n"
            "Try setting a lighter model with LEXAI_GEN_MODEL or verify internet/model cache."
        )
    return _TEXT_GEN


def _context_is_insufficient(context: str) -> bool:
    text = (context or "").strip()
    return len(text) < 120


def _structured_fallback(reason: str) -> str:
    return (
        "Law:\n"
        f"{reason}\n\n"
        "Penalty:\n"
        "Insufficient context in retrieved sources.\n\n"
        "Why this rule exists:\n"
        "Please retrieve more specific legal sections for reliable explanation.\n\n"
        "Practical Advice:\n"
        "Refine your query with act name and section number for better precision.\n\n"
        "Disclaimer:\n"
        "This is AI-generated legal information and not legal advice."
    )


def _build_prompt(context: str, question: str) -> str:
    return f"""You are a highly accurate legal expert assistant for Indian legal/compliance queries.
Use only the provided legal context.
If answer is not present, clearly say: "Insufficient context in retrieved sources."

Output MUST follow this exact structure:
Law:
Penalty:
Why this rule exists:
Practical Advice:
Disclaimer:

Context:
{context}

Question:
{question}

Answer:
"""


def _postprocess_output(generated_text: str) -> str:
    text = (generated_text or "").strip()
    required = ["Law:", "Penalty:", "Why this rule exists:", "Practical Advice:", "Disclaimer:"]
    if all(k in text for k in required):
        return text
    # Preserve content while enforcing response contract.
    return (
        "Law:\n"
        f"{text if text else 'Insufficient context in retrieved sources.'}\n\n"
        "Penalty:\n"
        "Based on retrieved context only; verify exact section and state-specific enforcement.\n\n"
        "Why this rule exists:\n"
        "To improve legal compliance and reduce risks.\n\n"
        "Practical Advice:\n"
        "Cross-check cited sections before acting.\n\n"
        "Disclaimer:\n"
        "This is AI-generated legal information and not legal advice."
    )


def generate_answer(context: str, question: str, style: str = "structured") -> str:
    """Generate answer from retrieved context and user question."""
    q = (question or "").strip()
    if not q:
        raise ValueError("question cannot be empty")

    ctx = (context or "").strip()
    if _context_is_insufficient(ctx):
        return _structured_fallback("Insufficient context in retrieved sources.")

    generator = _ensure_generator()
    prompt = _build_prompt(ctx, q)

    eos_id = None
    pad_id = None
    if _TOKENIZER is not None:
        eos_id = _TOKENIZER.eos_token_id
        pad_id = _TOKENIZER.eos_token_id if _TOKENIZER.pad_token_id is None else _TOKENIZER.pad_token_id

    outputs = generator(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )
    raw = outputs[0]["generated_text"] if outputs else ""
    completion = raw[len(prompt) :].strip() if raw.startswith(prompt) else raw.strip()

    if style == "structured":
        return _postprocess_output(completion)
    return completion


def generator_status() -> Dict:
    ready = _GEN_INIT_ERROR is None and _TEXT_GEN is not None
    return {
        "ready": bool(ready),
        "error": _GEN_INIT_ERROR or "",
        "model": GEN_MODEL_NAME,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }


# Load once at import.
_load_generator()

