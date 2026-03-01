from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User legal question in natural language")
    style: Optional[str] = Field(
        default=None,
        description="Optional style hint: very short, short, normal, detailed",
    )
    word_limit: Optional[int] = Field(
        default=None,
        ge=30,
        le=400,
        description="Optional word-limit hint for response length",
    )


class Citation(BaseModel):
    citation_text: str
    act_name: Optional[str] = None
    section: Optional[str] = None


class EvidenceItem(BaseModel):
    act: Optional[str] = None
    section: Optional[str] = None
    source_group: Optional[str] = None
    score: Optional[float] = None
    snippet: Optional[str] = None


class LegalAnswerResponse(BaseModel):
    query: str
    answer: str
    sections: List[str]
    citations: List[Citation]
    mode: str
    retrieval_source: str
    confidence: str
    preferences: Dict
    latency_ms: Dict[str, float]
    latency_profile: Dict
    evidence: List[EvidenceItem]


class HealthResponse(BaseModel):
    ok: bool
    status: str
    details: Dict
