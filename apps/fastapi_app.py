import re
from typing import Dict, List

from fastapi import FastAPI, HTTPException

from apps.lexai06_notebook_adapter import get_engine
from apps.schemas import AskRequest, Citation, EvidenceItem, HealthResponse, LegalAnswerResponse


app = FastAPI(
    title="LexAI 06 Legal QA API",
    version="1.0.0",
    description="Citation-rich legal QA API backed by notebook 06 retrieval/generation runtime.",
)


def _build_query_with_preferences(req: AskRequest) -> str:
    parts = [req.query.strip()]
    if req.style:
        style = req.style.strip().lower()
        if style in {"very short", "short", "normal", "detailed", "in detail", "in depth"}:
            parts.append(f"in {style}")
    if req.word_limit:
        parts.append(f"within {int(req.word_limit)} words")
    return " ".join([x for x in parts if x]).strip()


def _parse_citation(c: str) -> Citation:
    raw = str(c).strip()
    m = re.match(r"^(.*?)\s*-\s*Section\s*(.+)$", raw, flags=re.IGNORECASE)
    if not m:
        return Citation(citation_text=raw)
    act = m.group(1).strip() or None
    section = m.group(2).strip() or None
    return Citation(citation_text=raw, act_name=act, section=section)


def _normalize_response(user_query: str, payload: Dict) -> LegalAnswerResponse:
    sections = [str(x).strip() for x in payload.get("sections", []) if str(x).strip()]
    citations = [_parse_citation(x) for x in payload.get("citations", [])]
    evidence_rows: List[EvidenceItem] = []
    for e in payload.get("evidence", []) or []:
        evidence_rows.append(
            EvidenceItem(
                act=e.get("act"),
                section=e.get("section"),
                source_group=e.get("source_group"),
                score=float(e.get("score", 0.0)) if e.get("score") is not None else None,
                snippet=e.get("snippet"),
            )
        )

    return LegalAnswerResponse(
        query=user_query,
        answer=str(payload.get("answer", "")).strip(),
        sections=sections,
        citations=citations,
        mode=str(payload.get("mode", "unknown")),
        retrieval_source=str(payload.get("source", "unknown")),
        confidence=str(payload.get("confidence", "na")),
        preferences=payload.get("preferences", {}) or {},
        latency_ms=payload.get("latency_ms", {}) or {},
        latency_profile=payload.get("latency_profile", {}) or {},
        evidence=evidence_rows,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        engine = get_engine()
        details = engine.status()
        return HealthResponse(ok=bool(details.get("ready")), status="ready" if details.get("ready") else "cold", details=details)
    except Exception as exc:
        return HealthResponse(ok=False, status="error", details={"error": str(exc)})


@app.post("/v1/legal/answer", response_model=LegalAnswerResponse)
def answer(req: AskRequest) -> LegalAnswerResponse:
    query = _build_query_with_preferences(req)
    try:
        engine = get_engine()
        payload = engine.answer_query(query)
        return _normalize_response(user_query=req.query.strip(), payload=payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"LexAI engine unavailable: {exc}") from exc


@app.get("/v1/legal/latency")
def latency() -> Dict:
    try:
        engine = get_engine()
        return {"ok": True, "latency_profile": engine.latency_profile(), "status": engine.status()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Latency profile unavailable: {exc}") from exc
