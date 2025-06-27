from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Settings for the application"""
    # OpenAI API key
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Qdrant settings
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    collection_name: str = os.getenv("COLLECTION_NAME", "ragchatbot")

    # Hybrid search settings
    use_hybrid_search: bool = os.getenv("USE_HYBRID_SEARCH", "True").lower() == "true"
    hybrid_alpha: float = float(os.getenv("HYBRID_ALPHA", "0.5"))  # Weight for hybrid search

    # LLM settings
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))

    # Embedding settings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # OpenAI's small embedding model

    # Dimension for embeddings and vector store
    dimension: int = 1536  # Default dimension for OpenAI's ada-002 embeddings
    class Config:
        env_file = ".env"

settings = Settings()


