from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Settings for the application"""
    # OpenAI API key
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # LangSmith settings
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "rag-chatbot")
    langsmith_endpoint: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    langsmith_tracing: bool = os.getenv("LANGSMITH_TRACING", "True").lower() == "true"

    # Qdrant settings
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    collection_name: str = os.getenv("COLLECTION_NAME", "ragchatbot7")

    # Routing collection settings
    routing_collection_name: str = os.getenv("ROUTING_COLLECTION_NAME", "routing_questions")
    routing_similarity_threshold: float = float(os.getenv("ROUTING_SIMILARITY_THRESHOLD", "0.8"))

    # Hybrid search settings
    use_hybrid_search: bool = os.getenv("USE_HYBRID_SEARCH", "True").lower() == "true"
    hybrid_alpha: float = float(os.getenv("HYBRID_ALPHA", "0.5"))  # Weight for hybrid search

    # LLM settings
    model_name: str = os.getenv("MODEL_NAME")
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))

    # Embedding settings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")  # OpenAI's small embedding model

    # Dimension for embeddings and vector store
    dimension: int = int(os.getenv("DIMENSION", "3072"))  # Default for OpenAI's text-embedding-3-large

    # Google Drive integration settings
    google_drive_folder_id: str = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
    google_service_account_file: str = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service-account.json")
    scheduler_interval_hours: int = int(os.getenv("SCHEDULER_INTERVAL_HOURS", "1"))

    multi_model_api_key: str = os.getenv("MULTI_MODEL_API_KEY", "")
    multi_model_api_url: str = os.getenv("MULTI_MODEL_API_URL", "")

    class Config:
        env_file = ".env"

settings = Settings()
