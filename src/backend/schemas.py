from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ── Tenant ──────────────────────────────────────────────────────────────

class TenantCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


class TenantResponse(BaseModel):
    id: UUID
    name: str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Document ────────────────────────────────────────────────────────────

class DocumentUpload(BaseModel):
    tenant_id: UUID
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)


class DocumentResponse(BaseModel):
    id: UUID
    tenant_id: UUID
    title: str
    chunks_created: int
    created_at: datetime


# ── Ask ─────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    tenant_id: UUID
    question: str = Field(..., min_length=1, max_length=2000)


class AskAnswer(BaseModel):
    answer: Optional[str] = None
    citations: list[str] = []
    confidence: float = 0.0
    reason: Optional[str] = None
    cached: bool = False


# ── Health ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    database: str
    redis: str
