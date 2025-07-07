from typing import Dict, Optional, List, Tuple
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from app.config import settings
from app.routing_vector_store import RoutingVectorStore
import asyncio

logger = logging.getLogger(__name__)

class SmartQueryRouter:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.model_name,
            temperature=0.1,
            api_key=settings.openai_api_key
        )

        # Initialize routing vector store
        self.routing_vector_store = RoutingVectorStore()

        # Similarity threshold for routing decision
        self.similarity_threshold = settings.routing_similarity_threshold

        # LLM prompt for similarity evaluation
        self.similarity_prompt = """Bạn là một chuyên gia đánh giá độ tương đồng giữa các câu hỏi.

Nhiệm vụ: So sánh câu hỏi gốc với câu hỏi tìm được và đánh giá mức độ tương đồng về ý nghĩa.

Câu hỏi gốc: "{original_query}"

Câu hỏi tìm được: "{found_query}"

Hãy đánh giá mức độ tương đồng về ý nghĩa giữa 2 câu hỏi này theo thang điểm từ 0 đến 1:
- 0.9-1.0: Rất giống nhau về ý nghĩa, gần như tương đương
- 0.8-0.89: Tương đồng cao, có thể xử lý bằng cách tương tự
- 0.7-0.79: Tương đồng trung bình, có một số điểm chung
- 0.6-0.69: Tương đồng thấp, chỉ có ít điểm chung
- 0.0-0.59: Không tương đồng hoặc rất khác biệt

Chỉ trả về điểm số (ví dụ: 0.85), không giải thích thêm."""

        # Routing decision prompt
        self.routing_decision_prompt = """Bạn là bộ điều hướng thông minh của hệ thống chatbot FPT Assist.

Dựa trên thông tin sau, hãy quyết định điều hướng câu hỏi:

Câu hỏi gốc: "{original_query}"

Câu hỏi tương tự tìm được: "{similar_query}"
Độ tương đồng: {similarity_score}
Ngưỡng tương đồng: {threshold}

Quy tắc điều hướng:
- Nếu độ tương đồng >= {threshold}: Điều hướng đến RULE_BASED với câu hỏi tìm được
- Nếu độ tương đồng < {threshold}: Điều hướng đến RAG_CHAT với câu hỏi gốc

Trả về định dạng JSON:
{{
    "route": "RULE_BASED" hoặc "RAG_CHAT",
    "query": "câu hỏi được sử dụng để xử lý",
    "reason": "lý do điều hướng"
}}"""

    async def route_query(self, query: str) -> Dict[str, str]:
        """
        Điều hướng câu hỏi dựa trên tìm kiếm câu hỏi tương tự trong vector store

        Args:
            query (str): Câu hỏi gốc từ người dùng

        Returns:
            Dict với các key: route, query, reason, similarity_score (optional)
        """
        try:
            # Tìm kiếm câu hỏi tương tự trong vector store
            similar_results = await self.routing_vector_store.similarity_search_with_score(
                query, k=3
            )

            if not similar_results:
                logger.info("Không tìm thấy câu hỏi tương tự, điều hướng đến RAG_CHAT")
                return {
                    "route": "RAG_CHAT",
                    "query": query,
                    "reason": "Không tìm thấy câu hỏi tương tự trong cơ sở dữ liệu"
                }

            # Lấy câu hỏi tương tự nhất
            best_match_doc, vector_score = similar_results[0]
            similar_query = best_match_doc.page_content

            # Sử dụng LLM để đánh giá độ tương đồng semantic
            similarity_score = await self._evaluate_semantic_similarity(query, similar_query)

            # Quyết định điều hướng dựa trên độ tương đồng
            if similarity_score >= self.similarity_threshold:
                logger.info(f"Độ tương đồng cao ({similarity_score:.2f}), điều hướng đến RULE_BASED")
                return {
                    "route": "RULE_BASED",
                    "query": similar_query,
                    "reason": f"Tìm thấy câu hỏi tương tự với độ tương đồng {similarity_score:.2f}",
                    "similarity_score": similarity_score,
                    "original_query": query
                }
            else:
                logger.info(f"Độ tương đồng thấp ({similarity_score:.2f}), điều hướng đến RAG_CHAT")
                return {
                    "route": "RAG_CHAT",
                    "query": query,
                    "reason": f"Độ tương đồng thấp ({similarity_score:.2f}), không đủ để sử dụng rule-based",
                    "similarity_score": similarity_score,
                    "similar_query": similar_query
                }

        except Exception as e:
            logger.error(f"Error in smart query routing: {e}")
            # Fallback: điều hướng đến RAG_CHAT
            return {
                "route": "RAG_CHAT",
                "query": query,
                "reason": f"Lỗi trong quá trình điều hướng: {str(e)}"
            }

    async def _evaluate_semantic_similarity(self, query1: str, query2: str) -> float:
        """
        Đánh giá độ tương đồng semantic giữa 2 câu hỏi bằng LLM

        Args:
            query1 (str): Câu hỏi gốc
            query2 (str): Câu hỏi để so sánh

        Returns:
            float: Điểm tương đồng từ 0 đến 1
        """
        try:
            prompt = self.similarity_prompt.format(
                original_query=query1,
                found_query=query2
            )

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = response.content.strip()

            # Parse điểm số từ response
            try:
                score = float(result)
                return max(0.0, min(1.0, score))  # Đảm bảo score trong khoảng [0, 1]
            except ValueError:
                # Nếu không parse được, trả về 0.0
                logger.warning(f"Không thể parse điểm tương đồng: {result}")
                return 0.0

        except Exception as e:
            logger.error(f"Error evaluating semantic similarity: {e}")
            return 0.0

    async def add_routing_questions(self, questions_data: List[Dict]) -> bool:
        """
        Thêm câu hỏi vào routing vector store

        Args:
            questions_data (List[Dict]): Danh sách câu hỏi với format:
                [{"question": "câu hỏi", "category": "danh mục", "answer": "câu trả lời"}, ...]

        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            # Chuyển đổi sang Document objects
            documents = []
            for item in questions_data:
                doc = Document(
                    page_content=item["question"],
                    metadata={
                        "category": item.get("category", "unknown"),
                        "source": "routing_questions"
                    }
                )
                documents.append(doc)

            # Thêm vào vector store
            result = await self.routing_vector_store.add_questions(documents)
            logger.info(f"Đã thêm {len(documents)} câu hỏi vào routing vector store")
            return len(result) > 0

        except Exception as e:
            logger.error(f"Error adding routing questions: {e}")
            return False

    async def get_routing_stats(self) -> Dict:
        """
        Lấy thống kê về routing vector store

        Returns:
            Dict: Thông tin thống kê
        """
        try:
            collection_info = await self.routing_vector_store.get_collection_info()
            if collection_info:
                return {
                    "collection_name": settings.routing_collection_name,
                    "total_questions": collection_info.points_count,
                    "similarity_threshold": self.similarity_threshold,
                    "status": "active"
                }
            return {"status": "error", "message": "Không thể lấy thông tin collection"}
        except Exception as e:
            logger.error(f"Error getting routing stats: {e}")
            return {"status": "error", "message": str(e)}
