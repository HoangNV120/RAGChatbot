from typing import Dict, Optional, List, TypedDict, AsyncGenerator
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

class RAGChatStreaming:
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

        # Sử dụng singleton LLM để tránh tạo mới nhiều lần - WITH STREAMING
        if RAGChatStreaming._llm_instance is None:
            RAGChatStreaming._llm_instance = ChatOpenAI(
                model=settings.model_name,
                temperature=settings.temperature,
                api_key=settings.openai_api_key,
                max_retries=2,
                timeout=30,
                streaming=True,  # ENABLE STREAMING
            )
            # RAGChatStreaming._llm_instance = MultiModelChatAPI(
            #     api_key=settings.multi_model_api_key,
            #     model="gpt-4o-mini",
            #     api_url=settings.multi_model_api_url,
            # )
        self.llm = RAGChatStreaming._llm_instance

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

        # Khởi tạo LangGraph WITHOUT memory để debug (tương tự RAGChat)
        self.graph_app = None

    async def _ensure_graph_ready(self):
        """Đảm bảo graph đã được khởi tạo"""
        if self.graph_app is None:
            self.graph_app = await self._build_graph()

    async def _build_graph(self):
        """Build LangGraph WITHOUT memory checkpointer cho streaming - tương tự RAGChat"""
        builder = StateGraph(GraphState)

        # Node: truy xuất tài liệu với parallel processing
        async def retrieve(state: GraphState):
            question = state["messages"][-1].content
            subqueries = state.get('subqueries', [question])

            print(f"🔄 [STREAMING] Processing {len(subqueries)} subqueries")

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

        # Node: tạo câu trả lời từ LLM với streaming
        async def generate_stream(state: GraphState):
            question = state["messages"][-1].content
            docs = state["docs"]

            if docs:
                context_parts = []
                for doc in docs:
                    context_parts.append(doc.page_content)
                context = "\n\n".join(context_parts)
            else:
                context = "Không tìm thấy thông tin liên quan."

            prompt = self.prompt_template.format(
                system_prompt=self.system_prompt,
                context=context,
                question=question
            )

            print(f"🔄 [STREAMING] Context length: {len(context)} characters")

            try:
                # TRUE STREAMING với astream()
                full_content = ""
                async for chunk in self.llm.astream([HumanMessage(content=prompt)]):
                    if hasattr(chunk, 'content') and chunk.content:
                        full_content += chunk.content
                        # Yield streaming chunk ngay lập tức
                        yield {
                            "type": "stream_chunk",
                            "content": chunk.content,
                            "full_content": full_content,
                            "timestamp": time.time()
                        }

                # Cuối cùng yield complete response
                yield {
                    "type": "stream_complete",
                    "messages": state["messages"] + [HumanMessage(content=full_content)],
                    "docs": None
                }

            except asyncio.TimeoutError:
                logger.warning("LLM streaming timeout")
                yield {
                    "type": "stream_error",
                    "content": "Xin lỗi, hệ thống đang quá tải. Vui lòng thử lại sau."
                }

        # Xây đồ thị
        builder.add_node("retrieve", retrieve)
        builder.add_node("generate_stream", generate_stream)

        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "generate_stream")
        builder.add_edge("generate_stream", END)

        # Compile WITHOUT memory checkpointer (tương tự RAGChat)
        return builder.compile()

    async def generate_response_stream(self, query: str, session_id: Optional[str] = None) -> AsyncGenerator[Dict, None]:
        """
        TRUE STREAMING version với OpenAI astream() - BỎ QUA LANGGRAPH NODES
        """
        if not session_id or session_id.strip() == "":
            session_id = str(uuid4())

        try:
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

            # Bước 2: MANUAL RETRIEVAL (vì LangGraph không thể stream nodes)
            subqueries = rewrite_result.get("subqueries", [processed_query])
            k_per_query = max(1, min(3, 6 // len(subqueries)))

            print(f"🔄 [TRUE STREAMING] Processing {len(subqueries)} subqueries manually")

            try:
                # Parallel retrieval
                batch_result = await asyncio.wait_for(
                    self.vector_store.batch_similarity_search(subqueries, k=k_per_query),
                    timeout=15.0
                )

                all_results = batch_result["results"]  # ✅ Extract results from dictionary
                vector_metrics = batch_result["metrics"]  # ✅ Also get metrics for consistency

                # Combine results
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

                docs = combined_docs[:6]

            except Exception as e:
                logger.warning(f"Retrieval failed: {e}")
                docs = []

            # Bước 3: Build context
            if docs:
                context_parts = []
                for doc in docs:
                    context_parts.append(doc.page_content)
                context = "\n\n".join(context_parts)
            else:
                context = "Không tìm thấy thông tin liên quan."

            # Format prompt
            prompt = self.prompt_template.format(
                system_prompt=self.system_prompt,
                context=context,
                question=query
            )

            print(f"🔄 [TRUE STREAMING] Context length: {len(context)} characters")
            print(f"🔄 [TRUE STREAMING] Starting OpenAI astream()...")

            # Bước 4: TRUE OPENAI STREAMING với astream()
            full_response = ""
            try:
                async for chunk in self.llm.astream([HumanMessage(content=prompt)]):
                    if hasattr(chunk, 'content') and chunk.content:
                        full_response += chunk.content
                        # YIELD NGAY LẬP TỨC - TRUE STREAMING
                        yield {
                            "type": "chunk",
                            "content": chunk.content,
                            "timestamp": time.time()
                        }

            except Exception as e:
                logger.error(f"OpenAI streaming error: {e}")
                yield {
                    "type": "error",
                    "content": "🤖 Xin lỗi, có lỗi xảy ra. Bạn vui lòng thử lại sau.",
                    "timestamp": time.time()
                }
                return

            # Final done event
            yield {
                "type": "done",
                "session_id": session_id,
                "route_used": "RAG_CHAT_STREAMING_TRUE",
                "routing_info": {},
                "timestamp": time.time(),
                "subqueries": rewrite_result.get("subqueries", [processed_query]),
                "final_answer": full_response
            }

        except Exception as e:
            logger.error(f"Error in TRUE streaming: {e}")
            yield {
                "type": "error",
                "content": "🤖 Xin lỗi, có lỗi xảy ra. Bạn vui lòng thử lại sau.",
                "timestamp": time.time()
            }
