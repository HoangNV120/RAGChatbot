from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
import os
import asyncio
from typing import List
import time

class VectorStore:
    def __init__(self):
        # Initialize the embedding model (using OpenAI's small model)
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            dimensions=settings.dimension,
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
                        size=settings.dimension,  # Dimension for OpenAI's embeddings
                        distance=models.Distance.COSINE
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
            distance=models.Distance.COSINE,
            metadata_payload_key="metadata",
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
        print(f"🔍 Starting similarity search for query: '{query[:50]}{'...' if len(query) > 50 else ''}' (k={k})")
        total_start_time = time.time()

        # Đo thời gian embedding riêng
        print("🧠 Generating query embedding...")
        embedding_start_time = time.time()
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.embeddings.embed_query(query)
        )
        print(f"Query embedding shape: {len(query_embedding)}")
        embedding_end_time = time.time()
        embedding_time = embedding_end_time - embedding_start_time
        print(f"⏱️ Query embedding completed in {embedding_time:.3f}s")

        # Đo thời gian search riêng (không bao gồm embedding)
        print("🔍 Searching in vector database...")
        search_start_time = time.time()
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.client.search(
                collection_name=settings.collection_name,
                query_vector=("dense", query_embedding),  # Specify vector name
                limit=k,
                with_payload=True,
            )
        )
        print(f"Search result points count: {len(result)}")
        search_end_time = time.time()
        search_time = search_end_time - search_start_time
        print(f"⏱️ Vector search completed in {search_time:.3f}s")

        # Chuyển đổi kết quả về format Document
        from langchain.schema import Document
        documents = []
        for point in result:  # result is already a list of ScoredPoint
            content = point.payload.get('page_content', '')
            metadata = point.payload.get('metadata', {})
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        print(f"🎯 Total similarity search completed in {total_time:.3f}s (Embedding: {embedding_time:.3f}s + Search: {search_time:.3f}s) - Found {len(documents)} documents")

        return documents

    async def similarity_search_with_score(self, query, k=2):
        """
        Search for similar documents in the vector store and return scores

        Args:
            query (str): The query text
            k (int): Number of documents to retrieve

        Returns:
            List of tuples (document, score)
        """
        print(f"🔍 Starting similarity search with score for query: '{query[:50]}{'...' if len(query) > 50 else ''}' (k={k})")
        total_start_time = time.time()

        # Đo thời gian embedding riêng
        print("🧠 Generating query embedding...")
        embedding_start_time = time.time()
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.embeddings.embed_query(query)
        )
        embedding_end_time = time.time()
        embedding_time = embedding_end_time - embedding_start_time
        print(f"⏱️ Query embedding completed in {embedding_time:.3f}s")

        # Đo thời gian search riêng (không bao gồm embedding)
        print("🔍 Searching in vector database with scores...")
        search_start_time = time.time()
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.client.search(
                collection_name=settings.collection_name,
                query_vector=("dense", query_embedding),  # Specify vector name
                limit=k,
                with_payload=True,
                score_threshold=None
            )
        )
        search_end_time = time.time()
        search_time = search_end_time - search_start_time
        print(f"⏱️ Vector search completed in {search_time:.3f}s")

        # Chuyển đổi kết quả về format (Document, score)
        from langchain.schema import Document
        documents_with_scores = []
        for point in result:  # result is already a list of ScoredPoint
            content = point.payload.get('page_content', '')
            metadata = point.payload.get('metadata', {})
            doc = Document(page_content=content, metadata=metadata)
            score = point.score
            documents_with_scores.append((doc, score))

        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        print(f"🎯 Total similarity search with score completed in {total_time:.3f}s (Embedding: {embedding_time:.3f}s + Search: {search_time:.3f}s) - Found {len(documents_with_scores)} documents")

        return documents_with_scores

    async def add_documents(self, documents):
        """
        Add documents to the vector store

        Args:
            documents (List): List of documents to add

        Returns:
            IDs of the added documents
        """
        print(f"📄 Starting to add {len(documents)} documents to vector store")
        start_time = time.time()

        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.vector_store.add_documents(documents)
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"⏱️ Document addition completed in {elapsed_time:.3f}s - Generated embeddings for {len(documents)} documents")

        return result

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
    async def batch_similarity_search(self, queries: List[str], k: int = 2):
        """
        Parallel search for multiple queries with batch embedding
        """
        print(f"🔍 Starting batch similarity search for {len(queries)} queries (k={k})")
        total_start_time = time.time()

        # Batch embedding - TẤT CẢ queries cùng lúc
        print("🧠 Generating batch embeddings...")
        embedding_start_time = time.time()

        # Sử dụng embed_documents để batch tất cả queries
        embeddings = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.embeddings.embed_documents(queries)
        )

        embedding_end_time = time.time()
        embedding_time = embedding_end_time - embedding_start_time
        print(f"⏱️ Batch embedding completed in {embedding_time:.3f}s for {len(queries)} queries")

        # Parallel search với pre-computed embeddings
        print("🔍 Parallel searching in vector database...")
        search_start_time = time.time()

        search_tasks = []
        for i, (query, embedding) in enumerate(zip(queries, embeddings)):
            task = asyncio.create_task(
                self._search_with_precomputed_embedding(query, embedding, k, i)
            )
            search_tasks.append(task)

        # Chạy tất cả searches song song
        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        search_end_time = time.time()
        search_time = search_end_time - search_start_time
        print(f"⏱️ Parallel search completed in {search_time:.3f}s")

        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        print(f"🎯 Total batch search completed in {total_time:.3f}s (Embedding: {embedding_time:.3f}s + Search: {search_time:.3f}s)")

        return results

    async def _search_with_precomputed_embedding(self, query: str, embedding: List[float], k: int, query_index: int):
        """
        Search using precomputed embedding
        """
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.search(
                    collection_name=settings.collection_name,
                    query_vector=("dense", embedding),
                    limit=k,
                    with_payload=True,
                )
            )

            # Convert to Document objects
            from langchain.schema import Document
            documents = []
            for point in result:
                content = point.payload.get('page_content', '')
                metadata = point.payload.get('metadata', {})
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

            print(f"Query {query_index + 1}: Found {len(documents)} documents")
            return documents

        except Exception as e:
            print(f"Error in search {query_index + 1}: {e}")
            return []