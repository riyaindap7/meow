from functools import lru_cache
from typing import Optional
import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API
    APP_NAME: str = "VICTOR API"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = True

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"

    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    
    # MongoDB Configuration (replaces Supabase)
    # Note: For Docker, this will be mongodb://admin:admin123@mongodb:27017/
    MONGODB_URI: str = "mongodb://localhost:27017/"
    MONGODB_DATABASE: str = "victor_rag"

    # Google Drive
    GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE: Optional[str] = None
    GOOGLE_DRIVE_MASTER_FOLDER_ID: Optional[str] = None

    # Local storage
    LOCAL_STORAGE_ROOT: str = "backend/data/local_storage"

    # LLM
    OPENROUTER_API_KEY: Optional[str] = None
    LLM_MODEL: str = "alibaba/tongyi-deepresearch-30b-a3b:free"

    # Embeddings
    EMBEDDING_MODEL: str = "BAAI/bge-m3"

    # Document processing
    CHUNK_SIZE: int = 1000

    # ElevenLabs Configuration
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
