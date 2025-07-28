from typing import Dict, Optional
import logging
import time
import asyncio

from app.smart_router import SmartQueryRouter
from app.rag_chat import RAGChat
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

        # Khởi tạo router
        self.router = SmartQueryRouter()

        # Khởi tạo RAG Chat
        self.rag_chat = RAGChat(vector_store=vector_store)

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
            logger.info(f"Smart routing result: {routing_result}")
            print(f"Query routed to: {route}")
            print(f"Routing reason: {routing_result.get('reason', 'N/A')}")
            if 'similarity_score' in routing_result:
                print(f"Similarity score: {routing_result['similarity_score']:.3f}")

            if route == "DIRECT_ANSWER":
                # Trả về answer trực tiếp từ metadata khi score >= 80%
                direct_answer = routing_result.get("answer", "")
                return {
                    "output": direct_answer,
                    "session_id": session_id or "direct-answer-session",
                    "route_used": "DIRECT_ANSWER",
                    "routing_info": routing_result
                }
            else:
                # Sử dụng RAG chat với câu hỏi gốc
                query_to_use = routing_result["query"]  # Câu hỏi gốc
                result = await self.rag_chat.generate_response(query_to_use, session_id)
                result["route_used"] = "RAG_CHAT"
                result["routing_info"] = routing_result
                return result

        except Exception as e:
            logger.error(f"Error in master chatbot: {e}")
            return {
                "output": "🤖 Xin lỗi, có lỗi xảy ra. Bạn vui lòng thử lại sau.",
                "session_id": session_id or "error-session",
                "route_used": "ERROR"
            }

    async def generate_response_stream(self, query: str, session_id: Optional[str] = None):
        """
        Phương thức streaming để xử lý query và trả về từng chunk - chỉ stream chunk, done, error
        """
        try:
            # Routing (không stream)
            routing_result = await self.router.route_query(query)
            route = routing_result["route"]

            if route == "DIRECT_ANSWER":
                # Trả về answer trực tiếp từ metadata
                direct_answer = routing_result.get("answer", "")

                # Stream answer theo chunks
                words = direct_answer.split()
                chunk_size = 5  # 5 words per chunk

                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i + chunk_size])
                    yield {
                        "type": "chunk",
                        "content": chunk + (" " if i + chunk_size < len(words) else ""),
                        "route_used": "DIRECT_ANSWER",
                        "timestamp": time.time()
                    }
                    await asyncio.sleep(0.05)  # Small delay for streaming effect

                yield {
                    "type": "done",
                    "session_id": session_id or "direct-answer-session",
                    "route_used": "DIRECT_ANSWER",
                    "routing_info": routing_result,
                    "timestamp": time.time()
                }

            else:
                # RAG processing
                query_to_use = routing_result["query"]

                # Stream RAG response
                async for chunk in self.rag_chat.generate_response_stream(query_to_use, session_id):
                    chunk["route_used"] = "RAG_CHAT"
                    chunk["routing_info"] = routing_result
                    yield chunk

        except Exception as e:
            logger.error(f"Error in master chatbot stream: {e}")
            yield {
                "type": "error",
                "content": "🤖 Xin lỗi, có lỗi xảy ra. Bạn vui lòng thử lại sau.",
                "timestamp": time.time()
            }
