from typing import Dict, Optional, List, TypedDict
from uuid import uuid4
import logging
import asyncio
import time
from functools import lru_cache

from langchain_core.messages import HumanMessage, BaseMessage
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from app.vector_store import VectorStore
from app.pre_retrieval import PreRetrieval
from app.post_retrieval import PostRetrieval
from app.config import settings
from app.MultiModelChatAPI import MultiModelChatAPI

# Cấu hình logging với level cao hơn để giảm overhead
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Khai báo state cho LangGraph
class GraphState(TypedDict):
    messages: List[BaseMessage]
    docs: Optional[List]
    subqueries: Optional[List[str]]

class RAGChat:
    _llm_instance = None   # Singleton LLM instance

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store if vector_store else VectorStore()
        self.query_rewriter = PreRetrieval()
        self.post_retrieval = PostRetrieval()

        # Optimized system prompt using structured approach
        self.system_prompt = """<ROLE>Trợ lý Sinh viên FPTU</ROLE>

<CORE_RULES>
• ONLY use information from provided Context
• NO external knowledge beyond Context
• NO general queries (math, science, programming)
</CORE_RULES>

<RESPONSE_LOGIC>
IF Context contains sufficient info → Answer based on Context
IF Context has partial info → Answer partial + "Liên hệ Phòng Dịch Vụ Sinh Viên để biết thêm"
IF Context lacks info → "Mình chưa có dữ liệu, bạn liên hệ Phòng Dịch Vụ Sinh Viên nhé"
IF Context parts combine for clear conclusion → Provide logical conclusion
</RESPONSE_LOGIC>

<SPECIAL_CASES>
• Yes/No questions: "Đúng/Không đúng" + Context-based explanation
• Comparison/Verification: Compare with Context, correct if wrong
</SPECIAL_CASES>

<OUTPUT_STYLE>
• Use "bạn/mình" tone
• Quote directly from Context when possible
• Synthesize multiple Context parts if needed
• NO reverse questions
</OUTPUT_STYLE>"""

        # Sử dụng singleton LLM để tránh tạo mới nhiều lần - NO STREAMING
        if RAGChat._llm_instance is None:
            RAGChat._llm_instance = ChatOpenAI(
                model=settings.model_name,
                temperature=settings.temperature,
                api_key=settings.openai_api_key,
                max_retries=2,
                timeout=30,
                streaming=False,  # NO STREAMING
            )
            # RAGChat._llm_instance = MultiModelChatAPI(
            #     api_key=settings.multi_model_api_key,
            #     model_name="gpt-4.1-mini",
            #     api_url=settings.multi_model_api_url,
            # )
        self.llm = RAGChat._llm_instance

        # Optimized prompt template with clear structure and delimiters
        self.prompt_template = PromptTemplate.from_template(
            """{system_prompt}

<CONTEXT>
{context}
</CONTEXT>

<QUERY>
{question}
</QUERY>

<INSTRUCTIONS>
1. Analyze CONTEXT for relevant information
2. Check if CONTEXT contains answer to QUERY
3. Respond using RESPONSE_LOGIC above
</INSTRUCTIONS>

<RESPONSE>"""
        )

        # Khởi tạo LangGraph WITHOUT memory để debug
        self.graph_app = None

    async def _ensure_graph_ready(self):
        """Đảm bảo graph đã được khởi tạo"""
        if self.graph_app is None:
            self.graph_app = await self._build_graph()

    @lru_cache(maxsize=128)
    def _format_prompt_cached(self, system_prompt: str, context: str, question: str) -> str:
        """Cache prompt formatting để tránh format lại"""
        return self.prompt_template.format(
            system_prompt=system_prompt,
            context=context,
            question=question
        )

    async def _build_graph(self):
        """Build LangGraph WITHOUT memory checkpointer - chỉ để debug"""
        builder = StateGraph(GraphState)

        # Node: truy xuất tài liệu với parallel processing, subqueries và reranking
        async def retrieve(state: GraphState):
            question = state["messages"][-1].content
            subqueries = state.get('subqueries', [question])

            print(f"🔄 [DEBUG] Processing {len(subqueries)} subqueries")

            # Tối ưu k per query
            k_per_query = max(1, min(3, 6 // (len(subqueries))))

            try:
                # Sử dụng batch search
                all_results = await asyncio.wait_for(
                    self.vector_store.batch_similarity_search(subqueries, k=k_per_query),
                    timeout=15.0
                )

                # Kết hợp và loại bỏ trùng lặp
                combined_docs = []
                seen_content = set()

                for results in all_results:
                    if isinstance(results, list):
                        for doc in results:
                            content_hash = hash(doc.page_content[:200])
                            if content_hash not in seen_content:
                                seen_content.add(content_hash)
                                combined_docs.append(doc)
                                if len(combined_docs) >= 6:
                                    break

                return {**state, "docs": combined_docs[:6]}

            except asyncio.TimeoutError:
                logger.warning(f"Batch search timeout for subqueries")
                try:
                    docs = await asyncio.wait_for(
                        self.vector_store.similarity_search(question, k=4),
                        timeout=5.0
                    )
                    return {**state, "docs": docs}
                except:
                    return {**state, "docs": []}

        # Node: tạo câu trả lời từ LLM WITHOUT streaming
        async def generate(state: GraphState):
            question = state["messages"][-1].content
            docs = state["docs"]

            if docs:
                context_parts = []
                for doc in docs:
                    context_parts.append(doc.page_content)
                context = "\n\n".join(context_parts)
            else:
                context = "Không tìm thấy thông tin liên quan."

            prompt = self._format_prompt_cached(
                self.system_prompt,
                context,
                question
            )

            print(f"🔄 [DEBUG] Context length: {len(context)} characters")

            try:
                response = await asyncio.wait_for(
                    self.llm.ainvoke([HumanMessage(content=prompt)]),
                    timeout=25.0
                )
            except asyncio.TimeoutError:
                logger.warning("LLM response timeout")
                response = HumanMessage(content="Xin lỗi, hệ thống đang quá tải. Vui lòng thử lại sau.")

            return {
                "messages": state["messages"] + [response],
                "docs": None  # Clear docs after generation
            }

        # Xây đồ thị
        builder.add_node("retrieve", retrieve)
        builder.add_node("generate", generate)

        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)

        # Compile WITHOUT memory checkpointer
        return builder.compile()

    async def generate_response(self, query: str, session_id: Optional[str] = None) -> Dict:
        """Simple RAG response without memory and streaming"""
        await self._ensure_graph_ready()

        if not session_id or session_id.strip() == "":
            session_id = str(uuid4())

        # Bước 1: Phân tích và viết lại query
        try:
            rewrite_result = await asyncio.wait_for(
                self.query_rewriter.analyze_and_rewrite(query),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning("Query rewriter timeout, using original query")
            rewrite_result = {"can_process": True, "rewritten_query": query}

        processed_query = rewrite_result["rewritten_query"]

        # Bước 2: Chuẩn bị state với subqueries - NO MEMORY
        initial_state = {
            "messages": [HumanMessage(content=processed_query)],
            "subqueries": rewrite_result.get("subqueries", [processed_query])
        }

        # Bước 3: Gọi LangGraph WITHOUT memory
        try:
            result = await asyncio.wait_for(
                self.graph_app.ainvoke(initial_state),
                timeout=40.0
            )
            final_answer = result["messages"][-1].content
        except asyncio.TimeoutError:
            logger.warning("Graph execution timeout")
            final_answer = "Xin lỗi, hệ thống đang xử lý chậm. Vui lòng thử lại sau."

        return {
            "output": final_answer,
            "session_id": session_id,
            "subqueries": rewrite_result.get("subqueries", [processed_query])
        }
