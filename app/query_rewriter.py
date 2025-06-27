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
            api_key=settings.openai_api_key
        )

        self.rewrite_prompt = """Bạn là AI chuyên viết lại câu hỏi để tìm kiếm thông tin hiệu quả hơn.

Nhiệm vụ:
1. Phân tích câu hỏi gốc
2. Xác định xem có thể chia thành nhiều câu hỏi con không
3. Nếu CÓ THỂ CHIA (>2 chủ đề khác nhau), trả về: "KHÔNG XỬ LÝ"
4. Nếu KHÔNG THỂ CHIA (1 chủ đề chính), viết lại câu hỏi rõ ràng hơn

Ví dụ:
- "PL13 của K18 và điểm thi môn PRF192" → "KHÔNG XỬ LÝ" (2 chủ đề khác nhau)
- "Thông tin về PL13 K18" → "Thông tin chi tiết về chương trình học PL13 dành cho sinh viên khóa K18"
- "Cách đăng ký môn học và xem lịch thi" → "KHÔNG XỬ LÝ" (2 thao tác khác nhau)
- "Quy trình đăng ký môn học" → "Quy trình chi tiết để đăng ký môn học cho sinh viên FPTU"

Câu hỏi gốc: "{query}"

Trả về chỉ 1 trong 2:
- "KHÔNG XỬ LÝ" (nếu có >1 chủ đề)
- Câu hỏi đã được viết lại (nếu chỉ 1 chủ đề)
"""

        self.expansion_prompt = """Bạn là AI tạo ra các biến thể câu hỏi để tìm kiếm thông tin toàn diện.

Từ câu hỏi gốc, hãy tạo 2-3 câu hỏi liên quan để tìm kiếm thông tin đầy đủ hơn.

Câu hỏi gốc: "{query}"

Trả về các câu hỏi, mỗi câu một dòng:
"""

    async def analyze_and_rewrite(self, query: str) -> Dict[str, any]:
        """
        Phân tích và viết lại câu hỏi
        Returns: {
            "can_process": bool,
            "rewritten_query": str,
            "expanded_queries": List[str]
        }
        """
        try:
            # Bước 1: Kiểm tra có thể xử lý không
            rewrite_prompt = self.rewrite_prompt.format(query=query)
            response = await self.llm.ainvoke([HumanMessage(content=rewrite_prompt)])

            result = response.content.strip()

            if "KHÔNG XỬ LÝ" in result.upper():
                return {
                    "can_process": False,
                    "rewritten_query": query,
                    "expanded_queries": [query]
                }

            # Bước 2: Tạo các câu hỏi mở rộng
            expansion_prompt = self.expansion_prompt.format(query=result)
            expansion_response = await self.llm.ainvoke([HumanMessage(content=expansion_prompt)])

            expanded_queries = [
                line.strip()
                for line in expansion_response.content.strip().split('\n')
                if line.strip()
            ]

            # Đảm bảo có ít nhất câu hỏi gốc
            if not expanded_queries:
                expanded_queries = [result]

            return {
                "can_process": True,
                "rewritten_query": result,
                "expanded_queries": expanded_queries
            }

        except Exception as e:
            logger.error(f"Error in query rewriting: {e}")
            return {
                "can_process": True,
                "rewritten_query": query,
                "expanded_queries": [query]
            }

