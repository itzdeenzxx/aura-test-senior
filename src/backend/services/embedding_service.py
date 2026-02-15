import asyncio
import logging
from typing import Union

import google.generativeai as genai

from src.backend.config import settings

logger = logging.getLogger(__name__)

# Configure Gemini SDK once at module level
genai.configure(api_key=settings.GEMINI_API_KEY)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts.

    Uses asyncio.to_thread to avoid blocking the event loop.
    """
    def _embed() -> list[list[float]]:
        result = genai.embed_content(
            model=settings.EMBEDDING_MODEL,
            content=texts,
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=settings.EMBEDDING_DIMENSION,
        )
        return result["embedding"]

    logger.info("Generating embeddings for %d text(s)", len(texts))
    embeddings = await asyncio.to_thread(_embed)
    return embeddings


async def embed_query(text: str) -> list[float]:
    """Generate a single embedding for a query string."""
    def _embed() -> list[float]:
        result = genai.embed_content(
            model=settings.EMBEDDING_MODEL,
            content=text,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=settings.EMBEDDING_DIMENSION,
        )
        return result["embedding"]

    embedding = await asyncio.to_thread(_embed)
    return embedding
