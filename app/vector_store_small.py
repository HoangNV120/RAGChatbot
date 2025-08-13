from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
import os
import asyncio
import time

class VectorStoreSmall:
    """
    Vector Store cho collection ragsmall - chỉ embed cột question
    """
    def __init__(self):
        # Initialize the embedding model (same as main VectorStore)
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

        # Initialize Qdrant client
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )

        # Collection name cho ragsmall
        self.collection_name = "ragsmall"

        # Check if collection exists, if not create it
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            # Create collection with dense vectors only
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=3072,  # Dimension for text-embedding-3-large
                        distance=models.Distance.COSINE
                    )
                }
            )
            print(f"Created new Qdrant collection: {self.collection_name}")
        else:
            print(f"Qdrant collection '{self.collection_name}' already exists")

        # Initialize Qdrant vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            retrieval_mode=RetrievalMode.DENSE,
            vector_name="dense",
            content_payload_key="page_content",
            distance=models.Distance.COSINE
        )

    async def similarity_search(self, query, k=2):
        """
        Search for similar documents in the vector store

        Args:
            query (str): The query text
            k (int): Number of documents to retrieve

        Returns:
            List of relevant documents
        """
        # Run in executor to avoid blocking the event loop
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.vector_store.similarity_search(query, k=k)
        )

    async def similarity_search_with_score(self, query, k=2):
        """
        Search for similar documents in the vector store and return scores

        Args:
            query (str): The query text
            k (int): Number of documents to retrieve

        Returns:
            List of tuples (document, score)
        """
        # 🕐 VECTOR SEARCH START
        vector_search_start_time = time.time()
        print(f"🕐 [TIMER] VECTOR_SEARCH_WITH_SCORE START: {time.strftime('%H:%M:%S', time.localtime(vector_search_start_time))}")
        print(f"🔍 [RAGSMALL] Searching for query: '{query[:50]}...' with k={k}")

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.vector_store.similarity_search_with_score(query, k=k)
            )

            vector_search_end_time = time.time()
            vector_search_duration = vector_search_end_time - vector_search_start_time
            print(f"🕐 [TIMER] VECTOR_SEARCH_WITH_SCORE END: {time.strftime('%H:%M:%S', time.localtime(vector_search_end_time))} - Duration: {vector_search_duration:.3f}s")
            print(f"📊 [RAGSMALL] Found {len(result)} results")

            return result

        except Exception as e:
            vector_search_error_time = time.time()
            vector_search_error_duration = vector_search_error_time - vector_search_start_time
            print(f"🕐 [TIMER] VECTOR_SEARCH_WITH_SCORE ERROR: {time.strftime('%H:%M:%S', time.localtime(vector_search_error_time))} - Duration: {vector_search_error_duration:.3f}s")
            print(f"❌ [RAGSMALL] Error: {e}")
            raise

    async def add_documents(self, documents):
        """
        Add documents to the vector store

        Args:
            documents (List): List of documents to add

        Returns:
            IDs of the added documents
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.vector_store.add_documents(documents)
        )

    def get_collection_info(self):
        """
        Get information about the collection
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            return {"error": str(e)}

    def is_collection_empty(self):
        """
        Check if the collection is empty (no documents)
        
        Returns:
            bool: True if collection is empty, False if has documents
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count == 0
        except Exception as e:
            print(f"Error checking collection status: {e}")
            return True  # Assume empty if error occurs

    def collection_exists_and_has_data(self):
        """
        Check if collection exists and has data
        
        Returns:
            bool: True if collection exists and has data, False otherwise
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                return False
                
            info = self.client.get_collection(self.collection_name)
            return info.points_count > 0
        except Exception as e:
            print(f"Error checking collection existence and data: {e}")
            return False
