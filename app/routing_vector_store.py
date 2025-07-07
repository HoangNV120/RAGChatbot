from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
import asyncio
import logging

logger = logging.getLogger(__name__)

class RoutingVectorStore:
    def __init__(self):
        # Initialize the embedding model (using OpenAI's model)
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

        # Initialize Qdrant client
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )

        # Check if routing collection exists, if not create it
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if settings.routing_collection_name not in collection_names:
            # Create collection with dense vectors for routing questions
            self.client.create_collection(
                collection_name=settings.routing_collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=3072,  # Dimension for OpenAI's embeddings
                        distance=models.Distance.COSINE
                    )
                }
            )
            logger.info(f"Created new Qdrant collection: {settings.routing_collection_name}")

        # Initialize Qdrant vector store for routing questions
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=settings.routing_collection_name,
            embedding=self.embeddings,
            retrieval_mode=RetrievalMode.DENSE,
            vector_name="dense",
            content_payload_key="page_content",
            distance=models.Distance.COSINE
        )

    async def similarity_search_with_score(self, query: str, k: int = 5):
        """
        Search for similar questions in the routing vector store and return scores

        Args:
            query (str): The query text
            k (int): Number of questions to retrieve

        Returns:
            List of tuples (document, score)
        """
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.vector_store.similarity_search_with_score(query, k=k)
            )
        except Exception as e:
            logger.error(f"Error in routing similarity search: {e}")
            return []

    async def add_questions(self, questions_data):
        """
        Add questions to the routing vector store

        Args:
            questions_data (List): List of question documents to add
                Each document should contain: question text, category, answer, etc.

        Returns:
            IDs of the added documents
        """
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.vector_store.add_documents(questions_data)
            )
        except Exception as e:
            logger.error(f"Error adding questions to routing vector store: {e}")
            return []

    async def get_collection_info(self):
        """
        Get information about the routing collection
        """
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.get_collection(settings.routing_collection_name)
            )
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None
