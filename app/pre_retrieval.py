from typing import Dict
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from app.MultiModelChatAPI import MultiModelChatAPI
from app.config import settings

logger = logging.getLogger(__name__)

class PreRetrieval:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.model_name,
            temperature=0,  # Tăng temperature để có tính sáng tạo hơn cho query expansion
            api_key=settings.openai_api_key,
            max_tokens=300  # Tăng max_tokens để có thể sinh nhiều câu hỏi phụ hơn
        )
        # self.llm = MultiModelChatAPI(
        #     api_key=settings.multi_model_api_key,
        #     model_name="gpt-4.1-mini",
        #     api_url=settings.multi_model_api_url,
        # )

        # Optimized prompt using structured approach with clear delimiters
        self.subquery_prompt = """<ROLE>Query Analysis Expert</ROLE>

<TASK>Analyze the following question and decide how to process it:</TASK>

<INPUT_QUERY>
{query}
</INPUT_QUERY>

<PROCESSING_RULES>
COMPLEX Questions (generate EXACTLY 3 questions):
→ Questions with TWO OR MORE clearly distinct subjects/entities mentioned
→ Questions requiring COMPARISON between different subjects/entities  
→ Questions with multiple independent conditions in the same sentence
→ Examples: "A vs B", "A hay B", "A và B có gì khác nhau"

SIMPLE Questions (generate EXACTLY 1 question):
→ Questions about ONE main subject/entity (even with multiple attributes like rules, requirements, conditions)
→ Questions asking about conditions/requirements/rules/processes of a SINGLE subject
→ Direct factual questions about one specific thing
→ Questions with clear, single focus

IMPORTANT DISTINCTION:
- "Em cần điều kiện gì để đi thực tập?" → SIMPLE (one subject: thực tập)
- "KTX có luật lệ gì?" → SIMPLE (one subject: KTX)  
- "SWR302 hay FER202 lớn % hơn?" → COMPLEX (two subjects: SWR302 và FER202)
- "Nếu PRO192 được 4 điểm thì có học được PRJ301 không?" → COMPLEX (two subjects: PRO192 và PRJ301)
</PROCESSING_RULES>

<CONSTRAINTS>
• Complex questions → EXACTLY 3 questions (original + 2 focused sub-questions)
• Simple questions → EXACTLY 1 question (rewritten for clarity)
• Sub-questions must focus on specific aspects of the original query
</CONSTRAINTS>

<CLASSIFICATION_EXAMPLES>
COMPLEX (multiple distinct subjects/entities):
- "Assignment SWR302 hay FER202 lớn % hơn?" (SWR302 vs FER202)
- "Nếu tôi được 4 điểm PRO192 thì có được học PRJ301 không?" (PRO192 và PRJ301)
- "Học phí và lệ phí khác nhau như thế nào?" (học phí vs lệ phí)

SIMPLE (one main subject, even with complex attributes):
- "Em cần đáp ứng điều kiện gì để được đi thực tập?" (one subject: thực tập)
- "KTX có luật lệ gì cần nhớ hông?" (one subject: KTX)
- "Làm thế nào để đăng ký học phần?" (one subject: đăng ký học phần)
- "Học phí năm nay là bao nhiêu?" (one subject: học phí)
- "PRJ301 có những yêu cầu tiên quyết gì?" (one subject: PRJ301)
</CLASSIFICATION_EXAMPLES>

<EXAMPLES>
Complex: "Assignment SWR302 hay FER202 lớn % hơn?"
Output:
Assignment SWR302 hay FER202 lớn % hơn?
Assignment SWR302 chiếm bao nhiêu phần trăm tổng điểm?
Assignment FER202 chiếm bao nhiêu phần trăm tổng điểm?

Complex: "Nếu tôi được 4 điểm PRO192 thì có được học PRJ301 không?"
Output:
Nếu tôi được 4 điểm PRO192 thì có được học PRJ301 không?
Điều kiện tiên quyết để học môn PRJ301 là gì?
Môn PRO192 cần đạt điểm bao nhiêu để qua môn?

Simple: "Em cần đáp ứng điều kiện gì để được đi thực tập?"
Output:
Em cần đáp ứng điều kiện gì để được đi thực tập?

Simple: "KTX có luật lệ gì cần nhớ hông?"
Output:
KTX có luật lệ gì cần nhớ?

Simple: "Làm thế nào để đăng ký học phần?"
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
1. Count distinct subjects/entities in INPUT_QUERY
2. If 2+ distinct subjects/entities OR comparison → COMPLEX (3 questions)
3. If 1 main subject (regardless of complexity) → SIMPLE (1 question)
4. Generate questions following OUTPUT_FORMAT
</INSTRUCTIONS>"""

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
