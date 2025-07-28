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
            # model="text-embedding-3-large",
            api_key=settings.openai_api_key,
        )

        # Initialize sparse embeddings for hybrid search
        # self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

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
                        size=3072,  # Dimension for OpenAI's embeddings
                        distance=models.Distance.DOT
                    )
                },
                # sparse_vectors_config={
                #     "sparse": models.SparseVectorParams(
                #         index=models.SparseIndexParams(on_disk=False)
                #     )
                # }
            )
            print(f"Created new Qdrant collection: {settings.collection_name}")

        # Initialize Qdrant vector store with hybrid search
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=settings.collection_name,
            embedding=self.embeddings,
            # sparse_embedding=self.sparse_embeddings,
            # retrieval_mode=RetrievalMode.HYBRID,
            retrieval_mode=RetrievalMode.DENSE,
            vector_name="dense",
            # sparse_vector_name="sparse",
            content_payload_key="page_content",
            distance=models.Distance.DOT
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

    async def delete_documents_by_name(self, name):
        """
        Delete documents from the vector store based on metadata "name"

        Args:
            name (str): The name to match in metadata

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            # Get all documents with the specified name
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.name",
                        match=models.MatchValue(value=name)
                    )
                ]
            )

            # Search for documents to get their IDs
            search_result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.scroll(
                    collection_name=settings.collection_name,
                    scroll_filter=filter_condition,
                    limit=10000,  # Adjust based on your needs
                    with_payload=True,
                    with_vectors=False
                )
            )

            if not search_result[0]:  # No documents found
                print(f"No documents found with name: {name}")
                return False

            # Extract IDs from search results
            ids_to_delete = [point.id for point in search_result[0]]

            # Delete the documents
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.delete(
                    collection_name=settings.collection_name,
                    points_selector=models.PointIdsList(points=ids_to_delete)
                )
            )

            print(f"Successfully deleted {len(ids_to_delete)} documents with name: {name}")
            return True

        except Exception as e:
            print(f"Error deleting documents with name {name}: {str(e)}")
            return False

    async def update_documents_by_name(self, name, new_documents):
        """
        Update documents in the vector store based on metadata "name"
        This will delete existing documents with the name and add new ones

        Args:
            name (str): The name to match in metadata for deletion
            new_documents (List): List of new documents to add

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # First delete existing documents with the name
            delete_success = await self.delete_documents_by_name(name)

            if not delete_success:
                print(f"Failed to delete existing documents with name: {name}")
                # Continue with adding new documents even if deletion failed

            # Add new documents
            new_ids = await self.add_documents(new_documents)

            if new_ids:
                print(f"Successfully updated documents with name: {name}")
                print(f"Added {len(new_documents)} new documents")
                return True
            else:
                print(f"Failed to add new documents for name: {name}")
                return False

        except Exception as e:
            print(f"Error updating documents with name {name}: {str(e)}")
            return False

    async def get_documents_by_name(self, name):
        """
        Get all documents from the vector store based on metadata "name"

        Args:
            name (str): The name to match in metadata

        Returns:
            List of documents with the specified name
        """
        try:
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.name",
                        match=models.MatchValue(value=name)
                    )
                ]
            )

            # Search for documents
            search_result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.scroll(
                    collection_name=settings.collection_name,
                    scroll_filter=filter_condition,
                    limit=10000,
                    with_payload=True,
                    with_vectors=False
                )
            )

            documents = []
            if search_result[0]:
                for point in search_result[0]:
                    # Extract document content and metadata
                    content = point.payload.get('page_content', '')
                    metadata = point.payload.get('metadata', {})

                    # Create Document object
                    from langchain.schema import Document
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)

            print(f"Found {len(documents)} documents with name: {name}")
            return documents

        except Exception as e:
            print(f"Error getting documents with name {name}: {str(e)}")
            return []

    async def list_all_document_names(self):
        """
        Get a list of all unique document names in the vector store

        Returns:
            List of unique document names
        """
        try:
            # Get all documents
            search_result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.scroll(
                    collection_name=settings.collection_name,
                    limit=10000,
                    with_payload=True,
                    with_vectors=False
                )
            )

            names = set()
            if search_result[0]:
                for point in search_result[0]:
                    metadata = point.payload.get('metadata', {})
                    name = metadata.get('name')
                    if name:
                        names.add(name)

            names_list = list(names)
            print(f"Found {len(names_list)} unique document names")
            return names_list

        except Exception as e:
            print(f"Error listing document names: {str(e)}")
            return []
