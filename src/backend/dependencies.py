from uuid import UUID

from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import redis.asyncio as aioredis

from src.backend.config import settings
from src.backend.database import async_session
from src.backend.models import Tenant


async def get_db() -> AsyncSession:
    """Yield an async database session."""
    async with async_session() as session:
        yield session


async def get_redis() -> aioredis.Redis:
    """Yield a Redis client."""
    client = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


async def validate_tenant(
    tenant_id: UUID, db: AsyncSession
) -> Tenant:
    """Fail fast if tenant does not exist."""
    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = result.scalar_one_or_none()
    if tenant is None:
        raise HTTPException(status_code=404, detail=f"Tenant {tenant_id} not found")
    return tenant
