from typing import Dict, Optional
import logging

from app.category_partitioned_router import CategoryPartitionedRouter
from app.rag_chat import RAGChat
# from app.rule_based_chatbot import AdvancedChatbot  # TẮT TẠM THỜI
from app.safety_guard import SafetyGuard
from app.config import settings

# Loại bỏ basicConfig để tránh xung đột với main.py
logger = logging.getLogger(__name__)

class MasterChatbot:
    def __init__(self, vector_store=None):
        """
        Khởi tạo Master Chatbot
        """
        # Khởi tạo safety guard - kiểm tra đầu tiên
        self.safety_guard = SafetyGuard()

        # Khởi tạo category-partitioned router với pre-categorized data
        self.router = CategoryPartitionedRouter(use_categorized_data=True)

        # KÍCH HOẠT LẠI RAG Chat để xử lý category KHÁC và fallback
        self.rag_chat = RAGChat(vector_store=vector_store)

        # TẮT TẠM THỜI Rule-based chatbot
        # self.rule_based_chatbot = AdvancedChatbot()

    async def generate_response(self, query: str, session_id: Optional[str] = None) -> Dict:
        """
        Phương thức chính để xử lý query
        """
        try:
            # Bước 1: Kiểm tra tính an toàn của query
            # safety_result = await self.safety_guard.check_safety(query)
            #
            # if not safety_result["is_safe"]:
            #     logger.warning(f"Unsafe query detected: {query}")
            #     return {
            #         "output": safety_result["reason"],
            #         "session_id": session_id or "safety-blocked-session",
            #         "route_used": "SAFETY_BLOCKED"
            #     }

            # Bước 2: Định tuyến query (chỉ khi an toàn)
            routing_result = await self.router.route_query(query)
            route = routing_result["route"]

            # Log routing information
            logger.info(f"Optimized routing result: {routing_result}")
            print(f"Query routed to: {route}")
            
            # Hiển thị thông tin classification
            if "classification" in routing_result:
                classification = routing_result["classification"]
                print(f"LLM Classification: {classification.get('category', 'N/A')}")
            
            # Hiển thị thông tin similarity và category match nếu có
            if "similarity_score" in routing_result and routing_result["similarity_score"] > 0:
                print(f"Vector similarity: {routing_result['similarity_score']:.3f}")
                
            if route == "VECTOR_BASED":
                matched_category = routing_result.get('matched_category', 'N/A')
                print(f"Matched in category: {matched_category}")
                print(f"Matched question: {routing_result.get('matched_question', 'N/A')[:50]}...")

            if route == "VECTOR_BASED":
                # Trả về câu trả lời trực tiếp từ vector similarity search
                return {
                    "output": routing_result["answer"],
                    "session_id": session_id or "vector-based-session",
                    "route_used": "VECTOR_BASED",
                    "routing_info": routing_result
                }
            elif route == "RAG_CHAT":
                # Sử dụng RAG chat cho category KHÁC hoặc vector similarity thấp trong optimized partition
                query_to_use = routing_result["query"]  # Câu hỏi gốc
                result = await self.rag_chat.generate_response(query_to_use, session_id)
                result["route_used"] = "RAG_CHAT"
                result["routing_info"] = routing_result
                return result
            else:
                # Fallback case
                return {
                    "output": "🤖 Xin lỗi, tôi chưa thể trả lời câu hỏi này. Bạn có thể hỏi về các chủ đề như: ngành học, quy chế thi, điểm số, học phí, dịch vụ sinh viên, hoặc cơ sở vật chất.",
                    "session_id": session_id or "fallback-session",
                    "route_used": "FALLBACK",
                    "routing_info": routing_result
                }

        except Exception as e:
            logger.error(f"Error in master chatbot: {e}")
            return {
                "output": "🤖 Xin lỗi, có lỗi xảy ra. Bạn vui lòng thử lại sau.",
                "session_id": session_id or "error-session",
                "route_used": "ERROR"
            }
