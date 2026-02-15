import logging
from uuid import UUID

import tiktoken
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.config import settings
from src.backend.models import Document, DocumentChunk
from src.backend.services.embedding_service import embed_texts

logger = logging.getLogger(__name__)

# Initialise tokenizer once
_enc = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str) -> list[str]:
    """Split text into chunks of 500-800 tokens with 100 token overlap.

    Strategy:
    - Tokenize entire text.
    - Walk through tokens in windows of CHUNK_MAX_TOKENS.
    - Step forward by (CHUNK_MAX_TOKENS - CHUNK_OVERLAP_TOKENS) each iteration.
    - Decode each window back to text.
    """
    tokens = _enc.encode(text)
    chunks: list[str] = []

    step = settings.CHUNK_MAX_TOKENS - settings.CHUNK_OVERLAP_TOKENS
    start = 0

    while start < len(tokens):
        end = min(start + settings.CHUNK_MAX_TOKENS, len(tokens))
        chunk_tokens = tokens[start:end]

        # Skip tiny trailing fragments
        if len(chunk_tokens) < settings.CHUNK_MIN_TOKENS // 2 and chunks:
            # Append remainder to last chunk instead
            last_tokens = _enc.encode(chunks[-1])
            chunks[-1] = _enc.decode(last_tokens + chunk_tokens)
            break

        chunks.append(_enc.decode(chunk_tokens))
        start += step

    if not chunks:
        chunks = [text]

    logger.info("Chunked document into %d chunk(s)", len(chunks))
    return chunks


async def ingest_document(
    db: AsyncSession,
    tenant_id: UUID,
    title: str,
    content: str,
) -> tuple[Document, int]:
    """Full ingestion pipeline: create doc → chunk → embed → store chunks."""

    # 1. Create document record
    document = Document(tenant_id=tenant_id, title=title, content=content)
    db.add(document)
    await db.flush()  # get document.id

    # 2. Chunk
    chunks = chunk_text(content)

    # 3. Generate embeddings
    embeddings = await embed_texts(chunks)

    # 4. Store chunks with embeddings
    chunk_records = []
    for chunk_text_str, embedding in zip(chunks, embeddings):
        chunk_record = DocumentChunk(
            document_id=document.id,
            tenant_id=tenant_id,
            chunk_text=chunk_text_str,
            embedding=embedding,
        )
        chunk_records.append(chunk_record)

    db.add_all(chunk_records)
    await db.commit()

    logger.info(
        "Ingested document '%s' with %d chunks for tenant %s",
        title,
        len(chunk_records),
        tenant_id,
    )

    return document, len(chunk_records)
