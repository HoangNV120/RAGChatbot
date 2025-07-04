from typing import Dict, List, Optional, Tuple
import logging
import asyncio
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from app.config import settings

logger = logging.getLogger(__name__)

class PostRetrieval:
    def __init__(self):
        # LLM cho LLM-based reranking (backup method)
        self.llm = ChatOpenAI(
            model=settings.model_name,
            # model="gpt-4.1-nano",
            temperature=0.1,  # Temperature thấp để có tính nhất quán cao trong reranking
            api_key=settings.openai_api_key,
            max_tokens=100  # Ít token vì chỉ cần trả về ranking scores
        )

        # CrossEncoder Reranker với Hugging Face model phù hợp cho tiếng Việt
        try:
            # Sử dụng model multilingual phù hợp cho tiếng Việt
            self.cross_encoder_model = HuggingFaceCrossEncoder(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"  # Model nhẹ, hỗ trợ multilingual
            )
            self.cross_encoder_reranker = CrossEncoderReranker(
                model=self.cross_encoder_model
                # Bỏ top_k từ constructor vì không được hỗ trợ
            )
            self.use_cross_encoder = True
            logger.info("CrossEncoder reranker initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize CrossEncoder reranker: {e}, falling back to LLM reranking")
            self.use_cross_encoder = False

        # Prompt cho LLM-based reranking (fallback)
        self.reranking_prompt = """Hãy đánh giá mức độ liên quan của từng đoạn văn bản với câu hỏi được cho.

Câu hỏi: "{query}"

Các đoạn văn bản:
{documents}

Yêu cầu:
1. Đánh giá mức độ liên quan của từng đoạn văn (0-100)
2. Trả về kết quả theo format: "ID: score"
3. Một dòng cho mỗi đoạn văn
4. Ưu tiên các đoạn có thông tin trực tiếp trả lời câu hỏi

Ví dụ:
1: 95
2: 78
3: 45

Hãy đánh giá:"""

    async def rerank_documents(self, query: str, documents: List[Document], top_k: int = 6) -> List[Document]:
        """
        Rerank documents dựa trên mức độ liên quan với query

        Args:
            query (str): Câu hỏi gốc
            documents (List[Document]): Danh sách documents cần rerank
            top_k (int): Số lượng documents tốt nhất cần trả về

        Returns:
            List[Document]: Danh sách documents đã được rerank
        """
        if not documents:
            return []

        if len(documents) <= top_k:
            return documents

        try:
            # Ưu tiên sử dụng CrossEncoder nếu có
            if self.use_cross_encoder:
                reranked_docs = await self._cross_encoder_rerank(query, documents, top_k)
            else:
                # Fallback sang LLM-based reranking
                reranked_docs = await self._llm_rerank(query, documents, top_k)

            return reranked_docs

        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            # Fallback: trả về documents gốc
            return documents[:top_k]

    async def _cross_encoder_rerank(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        """
        Sử dụng CrossEncoder để rerank documents
        """
        try:
            # Chạy reranking trong executor để tránh block event loop
            reranked_docs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.cross_encoder_reranker.compress_documents(
                    documents=documents,
                    query=query
                )
            )

            # Giới hạn số lượng documents trả về
            return reranked_docs[:top_k]

        except Exception as e:
            logger.warning(f"CrossEncoder reranking failed: {e}, falling back to LLM reranking")
            return await self._llm_rerank(query, documents, top_k)

    async def _llm_rerank(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        """
        Sử dụng LLM để rerank documents (fallback method)
        """
        # Chuẩn bị format documents cho prompt
        doc_texts = []
        for i, doc in enumerate(documents, 1):
            # Cắt ngắn content để tránh prompt quá dài
            content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            doc_texts.append(f"{i}: {content}")

        documents_text = "\n\n".join(doc_texts)

        # Tạo prompt
        prompt = self.reranking_prompt.format(
            query=query,
            documents=documents_text
        )

        # Gọi LLM với timeout
        response = await asyncio.wait_for(
            self.llm.ainvoke([HumanMessage(content=prompt)]),
            timeout=15.0
        )

        # Parse kết quả
        scores = self._parse_scores(response.content, len(documents))

        # Tạo list các tuple (document, score) và sort theo score
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Trả về top_k documents
        return [doc for doc, _ in doc_scores[:top_k]]

    def _parse_scores(self, response: str, num_docs: int) -> List[float]:
        """
        Parse scores từ response của LLM
        """
        scores = [0.0] * num_docs  # Default scores

        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line:
                try:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        doc_id = int(parts[0].strip()) - 1  # Convert to 0-based index
                        score = float(parts[1].strip())

                        if 0 <= doc_id < num_docs:
                            scores[doc_id] = score
                except (ValueError, IndexError):
                    continue

        return scores

    async def semantic_rerank(self, query: str, documents: List[Document], top_k: int = 6) -> List[Document]:
        """
        Rerank dựa trên semantic similarity (using CrossEncoder if available)
        """
        if self.use_cross_encoder:
            return await self._cross_encoder_rerank(query, documents, top_k)
        else:
            return documents[:top_k]

    async def hybrid_rerank(self, query: str, documents: List[Document], top_k: int = 6) -> List[Document]:
        """
        Kết hợp nhiều phương pháp reranking
        """
        if not documents:
            return []

        try:
            # Bước 1: CrossEncoder reranking (nếu có)
            if self.use_cross_encoder:
                cross_encoder_results = await self._cross_encoder_rerank(query, documents, min(top_k * 2, len(documents)))
            else:
                cross_encoder_results = documents

            # Bước 2: LLM reranking để double-check (chỉ khi có nhiều docs)
            if len(cross_encoder_results) > top_k:
                final_results = await self._llm_rerank(query, cross_encoder_results, top_k)
            else:
                final_results = cross_encoder_results[:top_k]

            return final_results

        except Exception as e:
            logger.error(f"Error in hybrid reranking: {e}")
            return documents[:top_k]

    async def llm_rerank(self, query: str, documents: List[Document], top_k: int = 6) -> List[Document]:
        """
        Rerank documents chỉ sử dụng LLM-based reranking

        Args:
            query (str): Câu hỏi gốc
            documents (List[Document]): Danh sách documents cần rerank
            top_k (int): Số lượng documents tốt nhất cần trả về

        Returns:
            List[Document]: Danh sách documents đã được rerank theo LLM
        """
        if not documents:
            return []

        if len(documents) <= top_k:
            return documents

        return await self._llm_rerank(query, documents, top_k)

    def calculate_relevance_score(self, query: str, document: Document) -> float:
        """
        Tính điểm relevance đơn giản dựa trên keyword matching
        """
        try:
            query_words = set(query.lower().split())
            doc_words = set(document.page_content.lower().split())

            # Tính Jaccard similarity
            intersection = len(query_words.intersection(doc_words))
            union = len(query_words.union(doc_words))

            if union == 0:
                return 0.0

            return intersection / union

        except Exception:
            return 0.0

    async def filter_irrelevant_documents(self, query: str, documents: List[Document],
                                        min_relevance: float = 0.1) -> List[Document]:
        """
        Lọc bỏ các documents không liên quan
        """
        if not documents:
            return []

        relevant_docs = []
        for doc in documents:
            relevance = self.calculate_relevance_score(query, doc)
            if relevance >= min_relevance:
                relevant_docs.append(doc)

        return relevant_docs if relevant_docs else documents  # Fallback to original if all filtered out

