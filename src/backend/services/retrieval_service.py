import logging
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    chunk_text: str
    distance: float


async def retrieve_relevant_chunks(
    db: AsyncSession,
    tenant_id: UUID,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Retrieve top-k chunks by cosine distance, filtered by tenant_id.

    pgvector <=> operator returns cosine distance in [0, 2].
    We only return chunks with distance < SIMILARITY_DISTANCE_THRESHOLD (0.25).
    """
    query = sql_text("""
        SELECT
            id,
            document_id,
            chunk_text,
            embedding <=> :query_vec AS distance
        FROM document_chunks
        WHERE tenant_id = :tenant_id
        ORDER BY embedding <=> :query_vec
        LIMIT :top_k
    """)

    result = await db.execute(
        query,
        {
            "query_vec": str(query_embedding),
            "tenant_id": str(tenant_id),
            "top_k": top_k,
        },
    )

    rows = result.fetchall()

    chunks = []
    for row in rows:
        dist = float(row.distance)
        logger.info("Chunk %s distance: %.4f (threshold: %.2f)", str(row.id)[:8], dist, settings.SIMILARITY_DISTANCE_THRESHOLD)
        if dist < settings.SIMILARITY_DISTANCE_THRESHOLD:
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(row.id),
                    document_id=str(row.document_id),
                    chunk_text=row.chunk_text,
                    distance=dist,
                )
            )

    logger.info(
        "Retrieved %d/%d chunks under distance threshold %.2f for tenant %s",
        len(chunks),
        len(rows),
        settings.SIMILARITY_DISTANCE_THRESHOLD,
        tenant_id,
    )

    return chunks
