from typing import Dict, List
import logging
from langchain_core.documents import Document
from app.config import settings
from app.routing_vector_store import RoutingVectorStore

logger = logging.getLogger(__name__)

class SmartQueryRouter:
    def __init__(self):
        # Initialize routing vector store
        self.routing_vector_store = RoutingVectorStore()

        # Similarity threshold for routing decision
        self.similarity_threshold = settings.routing_similarity_threshold

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

            # Lấy câu hỏi tương tự nhất và score trực tiếp từ vector search
            best_match_doc, vector_score = similar_results[0]
            similar_query = best_match_doc.page_content

            # Sử dụng trực tiếp vector score thay vì LLM evaluation
            similarity_score = vector_score

            # Kiểm tra nếu score >= 80% (0.8), lấy trực tiếp answer từ metadata
            if similarity_score >= 0.8:
                direct_answer = best_match_doc.metadata.get("answer", "")
                if direct_answer:
                    logger.info(f"Độ tương đồng rất cao ({similarity_score:.2f}), trả về answer trực tiếp")
                    return {
                        "route": "DIRECT_ANSWER",
                        "query": similar_query,
                        "answer": direct_answer,
                        "reason": f"Độ tương đồng rất cao ({similarity_score:.2f}), trả về answer trực tiếp từ metadata",
                        "similarity_score": similarity_score,
                        "original_query": query
                    }

            # Nếu similarity score < 80%, điều hướng đến RAG_CHAT
            logger.info(f"Độ tương đồng thấp ({similarity_score:.2f}), điều hướng đến RAG_CHAT")
            return {
                "route": "RAG_CHAT",
                "query": query,
                "reason": f"Độ tương đồng thấp ({similarity_score:.2f}), sử dụng RAG processing",
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
                        "answer": item.get("answer", ""),  # Thêm answer vào metadata
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
