import logging

from google.api_core import exceptions as google_exceptions
from openai import RateLimitError as OpenAIRateLimitError
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

import redis.asyncio as aioredis

from src.backend.config import settings
from src.backend.database import async_session, engine
from src.backend.dependencies import get_db, get_redis, validate_tenant
from src.backend.models import AIRequest, AIResponse, Base, Tenant
from src.backend.schemas import (
    AskAnswer,
    AskRequest,
    DocumentResponse,
    DocumentUpload,
    HealthResponse,
    TenantCreate,
    TenantResponse,
)
from src.backend.services.cache_service import get_cached_answer, set_cached_answer
from src.backend.services.document_service import ingest_document
from src.backend.services.embedding_service import embed_query
from src.backend.services.llm_service import generate_answer
from src.backend.services.retrieval_service import retrieve_relevant_chunks

# ── Logging ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Aura — Internal Knowledge Assistant",
    description="Multi-tenant RAG-based knowledge assistant API",
    version="1.0.0",
)


# ── Startup / Shutdown ──────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    logger.info("Starting up — initialising database…")

    async with engine.begin() as conn:
        await conn.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialised successfully.")


@app.on_event("shutdown")
async def on_shutdown():
    await engine.dispose()
    logger.info("Database engine disposed.")


# ── Health ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check(
    db: AsyncSession = Depends(get_db),
    redis_client: aioredis.Redis = Depends(get_redis),
):
    db_status = "ok"
    redis_status = "ok"

    try:
        await db.execute(sql_text("SELECT 1"))
    except Exception:
        db_status = "error"

    try:
        await redis_client.ping()
    except Exception:
        redis_status = "error"

    status = "healthy" if db_status == "ok" and redis_status == "ok" else "unhealthy"
    return HealthResponse(status=status, database=db_status, redis=redis_status)


# ── Tenants ─────────────────────────────────────────────────────────────

@app.post("/tenants", response_model=TenantResponse, status_code=201)
async def create_tenant(
    body: TenantCreate,
    db: AsyncSession = Depends(get_db),
):
    tenant = Tenant(name=body.name)
    db.add(tenant)
    await db.commit()
    await db.refresh(tenant)
    logger.info("Created tenant '%s' (%s)", tenant.name, tenant.id)
    return tenant


# ── Documents ───────────────────────────────────────────────────────────

@app.post("/documents", response_model=DocumentResponse, status_code=201)
async def upload_document(
    body: DocumentUpload,
    db: AsyncSession = Depends(get_db),
):
    # Validate tenant exists
    await validate_tenant(body.tenant_id, db)

    try:
        document, chunk_count = await ingest_document(
            db=db,
            tenant_id=body.tenant_id,
            title=body.title,
            content=body.content,
        )
    except google_exceptions.ResourceExhausted:
        raise HTTPException(status_code=429, detail="Gemini API rate limit exceeded. Please retry later.")

    return DocumentResponse(
        id=document.id,
        tenant_id=document.tenant_id,
        title=document.title,
        chunks_created=chunk_count,
        created_at=document.created_at,
    )


# ── Ask ─────────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskAnswer)
async def ask_question(
    body: AskRequest,
    db: AsyncSession = Depends(get_db),
    redis_client: aioredis.Redis = Depends(get_redis),
):

    # 1. Validate tenant
    await validate_tenant(body.tenant_id, db)

    # 2. Check cache
    cached = await get_cached_answer(redis_client, body.tenant_id, body.question)
    if cached:
        cached["cached"] = True
        return AskAnswer(**cached)

    # 3. Embed question
    try:
        query_embedding = await embed_query(body.question)
    except google_exceptions.ResourceExhausted:
        raise HTTPException(status_code=429, detail="Gemini API rate limit exceeded. Please retry later.")

    # 4. Retrieve top-5 chunks (tenant-filtered)
    chunks = await retrieve_relevant_chunks(
        db=db,
        tenant_id=body.tenant_id,
        query_embedding=query_embedding,
        top_k=5,
    )

    # 5. Threshold gate — refuse if no relevant chunks
    if not chunks:
        refusal = AskAnswer(
            answer=None,
            citations=[],
            confidence=0.0,
            reason="insufficient_context",
            cached=False,
        )

        # Log the refusal
        ai_req = AIRequest(tenant_id=body.tenant_id, question=body.question)
        db.add(ai_req)
        await db.flush()

        ai_resp = AIResponse(
            request_id=ai_req.id,
            answer_json=refusal.model_dump(),
        )
        db.add(ai_resp)
        await db.commit()

        return refusal

    # 6. Generate answer via LLM
    try:
        answer_dict = await generate_answer(chunks, body.question)
    except (google_exceptions.ResourceExhausted, OpenAIRateLimitError):
        raise HTTPException(status_code=429, detail="LLM API rate limit exceeded. Please retry later.")

    # 7. Log request + response
    ai_req = AIRequest(tenant_id=body.tenant_id, question=body.question)
    db.add(ai_req)
    await db.flush()

    ai_resp = AIResponse(
        request_id=ai_req.id,
        answer_json=answer_dict,
    )
    db.add(ai_resp)
    await db.commit()

    # 8. Cache result
    await set_cached_answer(redis_client, body.tenant_id, body.question, answer_dict)

    # 9. Return
    return AskAnswer(**answer_dict, cached=False)
