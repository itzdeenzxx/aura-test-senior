import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # --- Gemini (embedding) ---
    GEMINI_API_KEY: str

    # --- OpenRouter (LLM generation) ---
    OPENROUTER_API_KEY: str
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # --- Postgres ---
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "knowledge_db"
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432

    # --- Redis ---
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379

    # --- RAG constants ---
    EMBEDDING_DIMENSION: int = 768
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    GENERATION_MODEL: str = "google/gemma-3-12b-it:free"
    MAX_OUTPUT_TOKENS: int = 500
    SIMILARITY_DISTANCE_THRESHOLD: float = 0.35
    CACHE_TTL: int = 900  # seconds
    CHUNK_MIN_TOKENS: int = 500
    CHUNK_MAX_TOKENS: int = 800
    CHUNK_OVERLAP_TOKENS: int = 100
    MAX_CONTEXT_TOKENS: int = 3000  # truncate context sent to LLM

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    class Config:
        env_file = ".env"
        case_sensitive = True


def get_settings() -> Settings:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY is missing or not set. "
            "Copy .env.example to .env and set a valid key."
        )
    return Settings()


settings = get_settings()
