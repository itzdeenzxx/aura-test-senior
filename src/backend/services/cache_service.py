import hashlib
import json
import logging
from typing import Optional
from uuid import UUID

import redis.asyncio as aioredis

from src.backend.config import settings

logger = logging.getLogger(__name__)


def _cache_key(tenant_id: UUID, question: str) -> str:
    """Build cache key: tenant_id:sha256(question)."""
    question_hash = hashlib.sha256(question.strip().lower().encode()).hexdigest()
    return f"{tenant_id}:{question_hash}"


async def get_cached_answer(
    redis_client: aioredis.Redis,
    tenant_id: UUID,
    question: str,
) -> Optional[dict]:
    """Return cached answer dict or None."""
    key = _cache_key(tenant_id, question)
    raw = await redis_client.get(key)
    if raw:
        logger.info("Cache HIT for key %s", key[:40])
        return json.loads(raw)
    logger.info("Cache MISS for key %s", key[:40])
    return None


async def set_cached_answer(
    redis_client: aioredis.Redis,
    tenant_id: UUID,
    question: str,
    answer: dict,
) -> None:
    """Cache an answer with TTL."""
    key = _cache_key(tenant_id, question)
    await redis_client.set(key, json.dumps(answer), ex=settings.CACHE_TTL)
    logger.info("Cached answer with key %s (TTL=%ds)", key[:40], settings.CACHE_TTL)
