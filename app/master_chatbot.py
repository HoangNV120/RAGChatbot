from typing import Dict, Optional
import logging

from app.category_router import CategoryRouter
from app.rag_chat import RAGChat  # KÍCH HOẠT LẠI
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

        # Khởi tạo category router - thay thế smart router
        self.router = CategoryRouter()

        # KÍCH HOẠT LẠI RAG Chat để xử lý category KHÁC
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
            logger.info(f"Category routing result: {routing_result}")
            print(f"Query routed to: {route}")
            print(f"Category: {routing_result.get('category', 'N/A')}")
            if route == "CATEGORY_BASED":
                print(f"Matched question: {routing_result.get('matched_question', 'N/A')}")

            if route == "CATEGORY_BASED":
                # Trả về câu trả lời trực tiếp từ database theo category
                return {
                    "output": routing_result["answer"],
                    "session_id": session_id or "category-based-session",
                    "route_used": "CATEGORY_BASED",
                    "routing_info": routing_result
                }
            elif route == "RAG_CHAT":
                # Sử dụng RAG chat cho category KHÁC hoặc không tìm thấy match
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
