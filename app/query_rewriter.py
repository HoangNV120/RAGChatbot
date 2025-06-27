from typing import Dict, List, Optional
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.config import settings

logger = logging.getLogger(__name__)

class QueryRewriter:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.model_name,
            temperature=0.1,
            api_key=settings.openai_api_key,
            max_tokens=150  # Giới hạn tokens để giảm latency
        )

        # Prompt được tối ưu hóa - ngắn gọn hơn và hiệu quả hơn
        self.rewrite_prompt = """Phân tích và viết lại câu hỏi sau để tìm kiếm hiệu quả hơn.

Quy tắc:
- Nếu câu hỏi có >1 chủ đề khác nhau → trả về "SKIP"
- Nếu câu hỏi chỉ 1 chủ đề → viết lại rõ ràng hơn với từ khóa FPTU

Câu hỏi: "{query}"

Trả về chỉ 1 dòng:"""

    async def analyze_and_rewrite(self, query: str) -> Dict[str, any]:
        """
        Phân tích và viết lại câu hỏi với latency thấp
        """
        try:
            # Chỉ gọi LLM 1 lần thay vì 2 lần
            rewrite_prompt = self.rewrite_prompt.format(query=query)
            response = await self.llm.ainvoke([HumanMessage(content=rewrite_prompt)])

            result = response.content.strip()

            if "SKIP" in result.upper():
                return {
                    "can_process": False,
                    "rewritten_query": query,
                    "expanded_queries": [query]
                }

            return {
                "can_process": True,
                "rewritten_query": result,
                "expanded_queries": [result]  # Bỏ expanded queries để tiết kiệm thời gian
            }

        except Exception as e:
            logger.error(f"Error in query rewriting: {e}")
            return {
                "can_process": True,
                "rewritten_query": query,
                "expanded_queries": [query]
            }
