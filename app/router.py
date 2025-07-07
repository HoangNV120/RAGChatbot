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

        self.routing_prompt = """Bạn là "Bộ Điều Hướng" của hệ thống AI chatbot FPT Assist.  
Nhiệm vụ duy nhất của bạn: Phân loại *mỗi câu hỏi của người dùng* sang một trong hai luồng xử lý:

1. RULE_BASED – Luồng FAQ rule-based:
   → Chọn khi câu hỏi:
   • RÕ RÀNG, dễ hiểu, không mơ hồ
   • Hỏi về các chủ đề liên quan đến trường đại học như: học phí, môn học, nội quy, đăng ký môn, bảo lưu, đóng học phí, lịch nghỉ lễ, v.v.
   • Câu hỏi ĐƠN GIẢN, không phức tạp
   • Có thể trả lời chính xác bằng một đoạn văn bản cố định

2. RAG_CHAT – Luồng Retrieval-Augmented Generation:
   → Chọn khi câu hỏi:
   • KHÔNG RÕ RÀNG: chứa nhiều từ viết tắt, các ký tự lạ, không rõ ý nghĩa
   • PHỨC TẠP: chứa nhiều ý, cần phân tích hoặc tổng hợp thông tin
   • Yêu cầu xử lý sâu hơn: ví dụ, giải thích chuyên sâu, so sánh, thống kê, hoặc suy luận
   • Câu hỏi mơ hồ, khó hiểu hoặc không thuộc các chủ đề trường đại học cơ bản

⚠ Quy tắc quan trọng: 
- Nếu câu hỏi RÕ RÀNG + về trường đại học + ĐƠN GIẢN → RULE_BASED
- Nếu câu hỏi KHÔNG RÕ RÀNG hoặc PHỨC TẠP → RAG_CHAT

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
