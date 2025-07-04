from typing import Dict, Optional
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.config import settings

logger = logging.getLogger(__name__)

class QueryRouter:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.model_name,
            temperature=0.1,
            api_key=settings.openai_api_key
        )

        self.routing_prompt = """Bạn là một AI phân loại câu hỏi của sinh viên FPTU.

Quy tắc phân loại:
1. RULE_BASED: Dành cho các câu hỏi đơn giản, FAQ cơ bản như:
   - Câu hỏi về địa điểm (phòng học, tòa nhà)
   - Câu hỏi về các thủ tục hành chính
   - Câu hỏi về học phí cơ bản
   - Câu hỏi ngắn gọn, rõ ràng (< 15 từ)
   - Câu hỏi có thể trả lời bằng thông tin cố định

2. RAG_CHAT: Dành cho các câu hỏi phức tạp như:
   - Câu hỏi dài, nhiều chi tiết (> 15 từ)
   - Câu hỏi yêu cầu giải thích, phân tích
   - Câu hỏi liên quan đến môn học theo từ thường hoặc regex r'^[A-ZĐ]{{3}}\d{{3}}[a-z]?$'
 hoặc có từ "LUK" (ví dụ: MLN121 ,ENW492c, LUK2, toán học, ...), khung chương trình
   - Câu hỏi cần tìm kiếm thông tin trong tài liệu

Chỉ trả về: RULE_BASED hoặc RAG_CHAT

Câu hỏi: "{query}"
"""

    async def route_query(self, query: str) -> str:
        """
        Định tuyến câu hỏi đến hệ thống phù hợp
        Returns: "RULE_BASED" hoặc "RAG_CHAT"
        """
        try:
            prompt = self.routing_prompt.format(query=query)
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])

            result = response.content.strip().upper()
            if "RULE_BASED" in result:
                return "RULE_BASED"
            elif "RAG_CHAT" in result:
                return "RAG_CHAT"
            else:
                # Fallback: câu hỏi ngắn -> rule-based, dài -> RAG
                return "RULE_BASED" if len(query.split()) < 15 else "RAG_CHAT"

        except Exception as e:
            logger.error(f"Error in query routing: {e}")
            # Fallback logic
            return "RULE_BASED" if len(query.split()) < 15 else "RAG_CHAT"
