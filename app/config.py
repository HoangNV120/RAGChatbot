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

    # Chroma settings (local vector store)
    chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

    # Weaviate settings
    weaviate_url: str = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    weaviate_api_key: Optional[str] = os.getenv("WEAVIATE_API_KEY")
    weaviate_class_name: str = os.getenv("WEAVIATE_CLASS_NAME", "RAGChatbot")

    # PostgreSQL/pgvector settings
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_user: str = os.getenv("POSTGRES_USER", "root")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "12345678")
    postgres_db: str = os.getenv("POSTGRES_DB", "ragchatbot")

    # Hybrid search settings
    use_hybrid_search: bool = os.getenv("USE_HYBRID_SEARCH", "True").lower() == "true"
    hybrid_alpha: float = float(os.getenv("HYBRID_ALPHA", "0.5"))  # Weight for hybrid search

    # LLM settings
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))

    # Gemini API settings
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    use_gemini_for_reranking: bool = os.getenv("USE_GEMINI_FOR_RERANKING", "False").lower() == "true"
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # Embedding settings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")  # OpenAI's small embedding model

    # Dimension for embeddings and vector store
    dimension: int = 3072

    # Pinecone settings
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "pcsk_UdUWs_Bx3yRmYfELogdHCF6LAmU6sjDrUz3Etw26QQivfwVNyh4CkhJRwS7RyTRYcYSeR")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "ragchatbot3")

    @property
    def postgres_url(self) -> str:
        """Construct PostgreSQL connection URL"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    class Config:
        env_file = ".env"

settings = Settings()
