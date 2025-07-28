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

    async def delete_questions_by_source(self, source_name):
        """
        Delete questions from the routing vector store based on metadata "source"

        Args:
            source_name (str): The source name to match in metadata

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            # Get all documents with the specified source
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value=source_name)
                    )
                ]
            )

            # Search for documents to get their IDs
            search_result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.scroll(
                    collection_name=settings.routing_collection_name,
                    scroll_filter=filter_condition,
                    limit=10000,
                    with_payload=True,
                    with_vectors=False
                )
            )

            if not search_result[0]:  # No documents found
                logger.info(f"No routing questions found with source: {source_name}")
                return False

            # Extract IDs from search results
            ids_to_delete = [point.id for point in search_result[0]]

            # Delete the documents
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.delete(
                    collection_name=settings.routing_collection_name,
                    points_selector=models.PointIdsList(points=ids_to_delete)
                )
            )

            logger.info(f"Successfully deleted {len(ids_to_delete)} routing questions with source: {source_name}")
            return True

        except Exception as e:
            logger.error(f"Error deleting routing questions with source {source_name}: {str(e)}")
            return False

    async def update_questions_by_source(self, source_name, new_questions):
        """
        Update questions in the routing vector store based on metadata "source"
        This will delete existing questions with the source and add new ones

        Args:
            source_name (str): The source name to match in metadata for deletion
            new_questions (List): List of new question documents to add

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # First delete existing questions with the source
            delete_success = await self.delete_questions_by_source(source_name)

            if not delete_success:
                logger.info(f"No existing routing questions found with source: {source_name}")
                # Continue with adding new questions even if deletion failed

            # Add new questions
            new_ids = await self.add_questions(new_questions)

            if new_ids:
                logger.info(f"Successfully updated routing questions with source: {source_name}")
                logger.info(f"Added {len(new_questions)} new routing questions")
                return True
            else:
                logger.error(f"Failed to add new routing questions for source: {source_name}")
                return False

        except Exception as e:
            logger.error(f"Error updating routing questions with source {source_name}: {str(e)}")
            return False

    async def get_questions_by_source(self, source_name):
        """
        Get all questions from the routing vector store based on metadata "source"

        Args:
            source_name (str): The source name to match in metadata

        Returns:
            List of question documents with the specified source
        """
        try:
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value=source_name)
                    )
                ]
            )

            # Search for documents
            search_result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.scroll(
                    collection_name=settings.routing_collection_name,
                    scroll_filter=filter_condition,
                    limit=10000,
                    with_payload=True,
                    with_vectors=False
                )
            )

            questions = []
            if search_result[0]:
                for point in search_result[0]:
                    # Extract question content and metadata
                    content = point.payload.get('page_content', '')
                    metadata = point.payload.get('metadata', {})

                    # Create Document object
                    from langchain.schema import Document
                    doc = Document(page_content=content, metadata=metadata)
                    questions.append(doc)

            logger.info(f"Found {len(questions)} routing questions with source: {source_name}")
            return questions

        except Exception as e:
            logger.error(f"Error getting routing questions with source {source_name}: {str(e)}")
            return []

    async def list_all_sources(self):
        """
        Get a list of all unique sources in the routing vector store

        Returns:
            List of unique source names
        """
        try:
            # Get all questions
            search_result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.scroll(
                    collection_name=settings.routing_collection_name,
                    limit=10000,
                    with_payload=True,
                    with_vectors=False
                )
            )

            sources = set()
            if search_result[0]:
                for point in search_result[0]:
                    metadata = point.payload.get('metadata', {})
                    source = metadata.get('source')
                    if source:
                        sources.add(source)

            sources_list = list(sources)
            logger.info(f"Found {len(sources_list)} unique sources in routing vector store")
            return sources_list

        except Exception as e:
            logger.error(f"Error listing sources: {str(e)}")
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
