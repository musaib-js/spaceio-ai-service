from pydantic import BaseModel, Field
import os

class Settings(BaseModel):
    MONGO_URI: str = Field(
        default_factory=lambda: os.getenv("MONGO_URI", "mongodb://localhost:27017")
    )
    MONGO_DB: str = os.getenv("MONGO_DB", "spaces_rag")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    PINECONE_API_KEY: str = Field(
        default_factory=lambda: os.getenv("PINECONE_API_KEY", "")
    )
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "spaces-rag")
    PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
    PINECONE_INDEX_HOST: str = os.getenv("PINECONE_HOST", "index.pinecone.io")

    GEMINI_API_KEY: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    GEMINI_CHAT_MODEL: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
    GEMINI_EMBED_MODEL: str = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

    JWT_SECRET: str = Field(
        default_factory=lambda: os.getenv("JWT_SECRET", "dev-secret-change-me")
    )
    JWT_ALG: str = os.getenv("JWT_ALG", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    DOC_CHUNK_SIZE: int = int(os.getenv("DOC_CHUNK_SIZE", "1200"))
    DOC_CHUNK_OVERLAP: int = int(os.getenv("DOC_CHUNK_OVERLAP", "150"))
    TOP_K: int = int(os.getenv("TOP_K", "6"))