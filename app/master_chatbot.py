from typing import Dict, Optional
import logging

from app.router import QueryRouter
from app.rag_chat import RAGChat
from app.rule_based_chatbot import RuleBasedChatbot
from app.safety_guard import SafetyGuard
from app.config import settings

logger = logging.getLogger(__name__)

class MasterChatbot:
    def __init__(self, vector_store=None):
        """
        Khởi tạo Master Chatbot
        """
        # Khởi tạo safety guard - kiểm tra đầu tiên
        self.safety_guard = SafetyGuard()

        # Khởi tạo router
        self.router = QueryRouter()

        # Khởi tạo RAG Chat
        self.rag_chat = RAGChat(vector_store=vector_store)

        # Khởi tạo Rule-based chatbot - sử dụng class mới
        self.rule_based_chatbot = RuleBasedChatbot()

    async def generate_response(self, query: str, session_id: Optional[str] = None) -> Dict:
        """
        Phương thức chính để xử lý query
        """
        try:
            # # Bước 1: Kiểm tra tính an toàn của query
            # safety_result = await self.safety_guard.check_safety(query)
            #
            # if not safety_result["is_safe"]:
            #     logger.warning(f"Unsafe query detected: {query}")
            #     return {
            #         "output": safety_result["reason"],
            #         "session_id": session_id or "safety-blocked-session",
            #         "route_used": "SAFETY_BLOCKED"
            #     }
            #
            # # Bước 2: Định tuyến query (chỉ khi an toàn)
            # route = await self.router.route_query(query)
            # logger.info(f"Query routed to: {route}")
            #
            # if route == "RULE_BASED":
            #     # Sử dụng rule-based chatbot
            #     answer = await self.rule_based_chatbot.chatbot_response(query)
            #     return {
            #         "output": answer,
            #         "session_id": session_id or "rule-based-session",
            #         "route_used": "RULE_BASED"
            #     }
            # else:
            #     # Sử dụng RAG chat
            #     result = await self.rag_chat.generate_response(query, session_id)
            #     result["route_used"] = "RAG_CHAT"
            #     return result
            result = await self.rag_chat.generate_response(query, session_id)
            result["route_used"] = "RAG_CHAT"
            return result

        except Exception as e:
            logger.error(f"Error in master chatbot: {e}")
            return {
                "output": "🤖 Xin lỗi, có lỗi xảy ra. Bạn vui lòng thử lại sau.",
                "session_id": session_id or "error-session",
                "route_used": "ERROR"
            }
