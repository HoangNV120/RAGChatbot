from typing import Dict
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from app.MultiModelChatAPI import MultiModelChatAPI
from app.config import settings

logger = logging.getLogger(__name__)

class PreRetrieval:
    def __init__(self):
        # self.llm = ChatOpenAI(
        #     model=settings.model_name,
        #     temperature=0.3,  # Tăng temperature để có tính sáng tạo hơn cho query expansion
        #     api_key=settings.openai_api_key,
        #     max_tokens=300  # Tăng max_tokens để có thể sinh nhiều câu hỏi phụ hơn
        # )
        self.llm = MultiModelChatAPI(
            api_key=settings.multi_model_api_key,
            model_name="gpt-4o-mini",
            api_url=settings.multi_model_api_url,
        )

        # Optimized prompt using structured approach with clear delimiters
        self.subquery_prompt = """<ROLE>Query Analysis Expert</ROLE>

<TASK>Analyze the following question and decide how to process it:</TASK>

<INPUT_QUERY>
{query}
</INPUT_QUERY>

<PROCESSING_RULES>
COMPLEX Questions (multiple subjects/concepts/requires reasoning):
→ Generate EXACTLY 3 questions: original + 2 sub-questions
→ Each sub-question focuses on 1 specific aspect
→ Original question always comes first

SIMPLE Questions (single clear topic):
→ Generate EXACTLY 1 question (rewrite original for clarity)
→ Use standard keywords and formal terminology
</PROCESSING_RULES>

<CONSTRAINTS>
• Total questions: MAX 3
• Complex → EXACTLY 3 questions
• Simple → EXACTLY 1 question
</CONSTRAINTS>

<EXAMPLES>
Complex: "Nếu tôi được 4 điểm tổng kết môn PRO192 thì có được học môn PRJ301 không?"
Output:
Nếu tôi được 4 điểm tổng kết môn PRO192 thì có được học môn PRJ301 không?
Điều kiện tiên quyết để học môn PRJ301 là gì?
Môn PRO192 cần đạt điểm tổng kết bao nhiêu để qua môn?

Simple: "làm sao đăng ký học phần"
Output:
Làm thế nào để đăng ký học phần?
</EXAMPLES>

<OUTPUT_FORMAT>
• One question per line
• NO numbering, NO bullet points
• NO explanations or comments
• Pure questions only
</OUTPUT_FORMAT>

<INSTRUCTIONS>
1. Analyze INPUT_QUERY complexity
2. Apply appropriate PROCESSING_RULES
3. Generate questions following OUTPUT_FORMAT
</INSTRUCTIONS>

Process the query above:"""

    async def analyze_and_rewrite(self, query: str) -> Dict[str, any]:
        """
        Phân tích và tạo các câu hỏi phụ (subqueries) từ câu hỏi chính
        """
        try:
            # Gọi LLM để tạo subqueries
            subquery_prompt = self.subquery_prompt.format(query=query)
            response = await self.llm.ainvoke([HumanMessage(content=subquery_prompt)])

            result = response.content.strip()

            # Tách các câu hỏi phụ thành danh sách đơn giản
            subqueries = []
            lines = result.split('\n')

            for line in lines:
                line = line.strip()
                # Chỉ loại bỏ dòng trống, giữ lại tất cả dòng có nội dung
                if line:
                    # Loại bỏ số thứ tự ở đầu dòng nếu có (1., 2., 1), 2))
                    if len(line) > 2 and line[0].isdigit():
                        if line[1:3] in ['. ', ') ']:
                            line = line[3:].strip()
                        elif line[1] in ['.', ')']:
                            line = line[2:].strip()

                    # Loại bỏ dấu gạch đầu dòng nếu có
                    if line.startswith('- '):
                        line = line[2:].strip()
                    elif line.startswith('* '):
                        line = line[2:].strip()

                    # Thêm vào danh sách nếu còn nội dung sau khi clean
                    if line:
                        subqueries.append(line)

            # Nếu không tạo được câu hỏi phụ, sử dụng câu hỏi gốc
            if not subqueries:
                subqueries = [query]

            # Giới hạn tối đa 3 câu hỏi như đã quy định trong prompt
            subqueries = subqueries[:3]

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
