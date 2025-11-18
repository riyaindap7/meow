from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    APP_NAME: str = "VICTOR API"
    APP_VERSION: str = "2.0.0"  # Updated for local storage architecture
    DEBUG: bool = True
    
    # CORS Configuration
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"
    
    # Milvus Configuration
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    
    # MongoDB Configuration (replaces Supabase)
    MONGODB_URI: str = "mongodb://localhost:27017/"
    MONGODB_DATABASE: str = "victor_rag"
    
    # Google Drive Configuration
    GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE: Optional[str] = None
    GOOGLE_DRIVE_MASTER_FOLDER_ID: Optional[str] = None
    
    # Local Storage Configuration
    LOCAL_STORAGE_ROOT: str = "backend/data/local_storage"
    
    # OpenRouter Configuration (for LLM)
    OPENROUTER_API_KEY: Optional[str] = None
    LLM_MODEL: str = "meta-llama/llama-2-7b-chat:free"
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Document Processing Configuration
    CHUNK_SIZE: int = 1000
    
    class Config:
        env_file = ".env"
        case_sensitive = True

def get_settings() -> Settings:
    """Get application settings"""
    return Settings()
