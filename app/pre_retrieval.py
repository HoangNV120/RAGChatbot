from typing import Dict
import logging
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.callbacks.manager import get_openai_callback

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

<TASK>Analyze the following query and decide how to process it:</TASK>

<INPUT_QUERY>
{query}
</INPUT_QUERY>

<PROCESSING_RULES>
COMPLEX Queries (generate EXACTLY 3 queries):
→ Queries with TWO OR MORE clearly distinct subjects/entities mentioned
→ Queries requiring COMPARISON between different subjects/entities  
→ Queries with multiple independent conditions or requirements
→ Examples: "A vs B", "A hay B", "A và B có gì khác nhau", "So sánh A và B"

SIMPLE Queries (generate EXACTLY 1 query):
→ Queries about ONE main subject/entity (even with multiple attributes like rules, requirements, conditions)
→ Queries asking about conditions/requirements/rules/processes of a SINGLE subject
→ Direct factual queries about one specific thing
→ Queries with clear, single focus
→ Statements or requests about a single topic
</PROCESSING_RULES>

<CONSTRAINTS>
• Complex queries → EXACTLY 3 queries (original + 2 focused sub-queries)
• Simple queries → EXACTLY 1 query (rewritten for clarity)
• Sub-queries must focus on specific aspects of the original query
• Handle all types of queries: questions, statements, requests, commands
</CONSTRAINTS>

<CLASSIFICATION_EXAMPLES>
COMPLEX (multiple distinct subjects/entities):
- "Assignment SWR302 hay FER202 lớn % hơn?" (SWR302 vs FER202)
- "Nếu tôi được 4 điểm PRO192 thì có được học PRJ301 không?" (PRO192 và PRJ301)
- "Học phí và lệ phí khác nhau như thế nào?" (học phí vs lệ phí)
- "So sánh giữa KTX A và KTX B" (KTX A vs KTX B)
- "Tôi muốn biết về SWR302 và FER202" (SWR302 và FER202)

SIMPLE (one main subject, even with complex attributes):
- "Em cần đáp ứng điều kiện gì để được đi thực tập?" (one subject: thực tập)
- "KTX có luật lệ gì cần nhớ hông?" (one subject: KTX)
- "Làm thế nào để đăng ký học phần?" (one subject: đăng ký học phần)
- "Học phí năm nay là bao nhiêu?" (one subject: học phí)
- "PRJ301 có những yêu cầu tiên quyết gì?" (one subject: PRJ301)
- "Hướng dẫn đăng ký môn học" (one subject: đăng ký môn học)
- "Thông tin về học bổng" (one subject: học bổng)
- "Quy định về KTX" (one subject: KTX)
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

Complex: "Tôi muốn biết về SWR302 và FER202"
Output:
Tôi muốn biết về SWR302 và FER202
Thông tin chi tiết về môn SWR302
Thông tin chi tiết về môn FER202

Simple: "Em cần đáp ứng điều kiện gì để được đi thực tập?"
Output:
Em cần đáp ứng điều kiện gì để được đi thực tập?

Simple: "KTX có luật lệ gì cần nhớ hông?"
Output:
KTX có luật lệ gì cần nhớ?

Simple: "Hướng dẫn đăng ký môn học"
Output:
Hướng dẫn đăng ký môn học

Simple: "Thông tin về học bổng"
Output:
Thông tin về học bổng
</EXAMPLES>

<OUTPUT_FORMAT>
• One query per line
• NO numbering, NO bullet points
• NO explanations or comments
• Pure queries only
</OUTPUT_FORMAT>

<INSTRUCTIONS>
1. Count distinct subjects/entities in INPUT_QUERY
2. If 2+ distinct subjects/entities OR comparison → COMPLEX (3 queries)
3. If 1 main subject (regardless of complexity) → SIMPLE (1 query)
4. Generate queries following OUTPUT_FORMAT
5. Handle all query types: questions, statements, requests, commands
</INSTRUCTIONS>"""

    async def analyze_and_rewrite(self, query: str) -> Dict[str, any]:
        """
        Phân tích và tạo các câu hỏi phụ (subqueries) từ câu hỏi chính
        Sử dụng get_openai_callback để thu thập metrics chính xác
        """
        # Khởi tạo metrics
        metrics = {
            "pre_retrieval_time": 0,
            "pre_retrieval_cost": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "successful_requests": 0,
            "subqueries_count": 0,
            "errors": []
        }

        start_time = time.time()

        try:
            # Gọi LLM với callback để tạo subqueries
            subquery_prompt = self.subquery_prompt.format(query=query)

            # Sử dụng get_openai_callback để thu thập metrics
            with get_openai_callback() as cb:
                response = await self.llm.ainvoke([HumanMessage(content=subquery_prompt)])

                # Thu thập metrics từ callback
                metrics.update({
                    "pre_retrieval_cost": cb.total_cost,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_tokens": cb.total_tokens,
                    "successful_requests": cb.successful_requests,
                })

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

            # Cập nhật metrics
            end_time = time.time()
            metrics.update({
                "pre_retrieval_time": end_time - start_time,
                "subqueries_count": len(subqueries)
            })

            print(f"🧠 Pre-retrieval completed: {len(subqueries)} subqueries")
            print(f"   💰 Cost: ${metrics['pre_retrieval_cost']:.6f}")
            print(f"   ⏱️  Time: {metrics['pre_retrieval_time']:.3f}s")
            print(f"   📊 Tokens - Input: {metrics['prompt_tokens']}, Output: {metrics['completion_tokens']}, Total: {metrics['total_tokens']}")

            return {
                "can_process": True,
                "rewritten_query": subqueries[0],  # Câu hỏi chính để hiển thị
                "subqueries": subqueries,
                "metrics": metrics  # ✅ Truyền metrics từ callback
            }

        except Exception as e:
            end_time = time.time()
            metrics.update({
                "pre_retrieval_time": end_time - start_time,
                "subqueries_count": 1,
                "errors": [f"Pre-retrieval error: {str(e)}"]
            })

            logger.error(f"Error in creating subqueries: {e}")
            return {
                "can_process": True,
                "rewritten_query": query,
                "subqueries": [query],
                "metrics": metrics  # ✅ Truyền metrics ngay cả khi có lỗi
            }
