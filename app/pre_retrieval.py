from typing import Dict, List, Optional
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.config import settings

logger = logging.getLogger(__name__)

class PreRetrieval:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.model_name,
            temperature=0.3,  # Tăng temperature để có tính sáng tạo hơn cho query expansion
            api_key=settings.openai_api_key,
            max_tokens=300  # Tăng max_tokens để có thể sinh nhiều câu hỏi phụ hơn
        )

        # Prompt được tối ưu hóa cho tạo subqueries (hỗ trợ cả tiếng Việt và tiếng Anh)
        self.subquery_prompt = """Hãy phân tích câu hỏi sau và tạo ra từ 1 đến 3 câu hỏi phụ để tìm kiếm thông tin hiệu quả hơn.

Câu hỏi chính: "{query}"

Yêu cầu:
1. Giữ nguyên ý nghĩa chính của câu hỏi gốc
2. Tạo thêm các câu hỏi phụ với từ khóa khác nhau bằng tiếng Việt
3. Mở rộng ngữ cảnh liên quan
4. Mỗi câu hỏi phụ trên 1 dòng, không đánh số

Ví dụ:
Câu hỏi chính: "Làm thế nào để đăng ký học phần?"
Các câu hỏi phụ:
Làm thế nào để đăng ký học phần?
Quy trình đăng ký môn học như thế nào?

Hãy tạo các câu hỏi phụ cho câu hỏi trên:"""

    async def analyze_and_rewrite(self, query: str) -> Dict[str, any]:
        """
        Phân tích và tạo các câu hỏi phụ (subqueries) từ câu hỏi chính
        """
        try:
            # Gọi LLM để tạo subqueries
            subquery_prompt = self.subquery_prompt.format(query=query)
            response = await self.llm.ainvoke([HumanMessage(content=subquery_prompt)])

            result = response.content.strip()

            # Tách các câu hỏi phụ thành danh sách
            subqueries = []
            lines = result.split('\n')

            for line in lines:
                line = line.strip()
                # Loại bỏ các dòng trống và dòng không phải câu hỏi
                if (line and len(line) > 10 and
                    not line.startswith('Câu hỏi') and
                    not line.startswith('Các câu hỏi') and
                    line != "Các câu hỏi phụ:"):
                    # Loại bỏ số thứ tự nếu có
                    if line[0].isdigit() and line[1:3] in ['. ', ') ']:
                        line = line[3:]
                    elif line[0].isdigit() and line[1] in ['.', ')']:
                        line = line[2:]
                    subqueries.append(line.strip())

            # Nếu không tạo được câu hỏi phụ, sử dụng câu hỏi gốc
            if not subqueries:
                subqueries = [query]

            # Đảm bảo câu hỏi gốc luôn có trong danh sách
            if query not in subqueries:
                subqueries.insert(0, query)

            # Giới hạn tối đa 4 câu hỏi để tránh quá tải
            subqueries = subqueries[:4]

            return {
                "can_process": True,
                "rewritten_query": subqueries[0],  # Câu hỏi chính để hiển thị
                "subqueries": subqueries
            }

        except Exception as e:
            logger.error(f"Error in creating subqueries: {e}")
            return {
                "can_process": True,
                "rewritten_query": query,
                "subqueries": [query]
            }

