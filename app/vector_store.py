from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
import os
import asyncio

class VectorStore:
    def __init__(self):
        # Initialize the embedding model (using OpenAI's small model)
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key
        )

        # Initialize sparse embeddings for hybrid search
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        # Initialize Qdrant client
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )

        # Check if collection exists, if not create it with proper configuration for hybrid search
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if settings.collection_name not in collection_names:
            # Create collection with both dense and sparse vectors
            self.client.create_collection(
                collection_name=settings.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=1536,  # Dimension for OpenAI's embeddings
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                }
            )
            print(f"Created new Qdrant collection: {settings.collection_name}")

        # Initialize Qdrant vector store with hybrid search
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=settings.collection_name,
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
            content_payload_key="page_content"
        )

    async def similarity_search(self, query, k=4):
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

    async def similarity_search_with_score(self, query, k=4):
        """
        Search for similar documents in the vector store and return scores

        Args:
            query (str): The query text
            k (int): Number of documents to retrieve

        Returns:
            List of tuples (document, score)
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.vector_store.similarity_search_with_score(query, k=k)
        )

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


