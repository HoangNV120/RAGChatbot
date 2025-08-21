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
from langchain_community.callbacks.manager import get_openai_callback
from app.vector_store import VectorStore
from app.pre_retrieval import PreRetrieval
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
    vector_metrics: Optional[Dict]  # ✅ Thêm field cho vector metrics
    retrieved_docs: Optional[List]  # ✅ Thêm field cho retrieved docs info

class RAGChat:
    _llm_instance = None   # Singleton LLM instance

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store if vector_store else VectorStore()
        self.query_rewriter = PreRetrieval()

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
            #     model="gpt-4.1-mini",
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
                # Sử dụng batch search với metrics
                batch_result = await asyncio.wait_for(
                    self.vector_store.batch_similarity_search(subqueries, k=k_per_query),
                    timeout=15.0
                )

                all_results = batch_result["results"]
                vector_metrics = batch_result["metrics"]

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

                return {
                    **state,
                    "docs": combined_docs[:6],
                    "vector_metrics": vector_metrics  # ✅ Truyền metrics từ vector store
                }

            except asyncio.TimeoutError:
                logger.warning(f"Batch search timeout for subqueries")
                try:
                    # Fallback search với metrics
                    search_result = await asyncio.wait_for(
                        self.vector_store.similarity_search(question, k=4),
                        timeout=5.0
                    )
                    docs = search_result["documents"]
                    vector_metrics = search_result["metrics"]

                    return {
                        **state,
                        "docs": docs,
                        "vector_metrics": vector_metrics  # ✅ Truyền metrics từ fallback
                    }
                except:
                    return {
                        **state,
                        "docs": [],
                        "vector_metrics": {
                            "rag_embedding_time": 0,
                            "rag_vector_search_time": 0,
                            "rag_total_search_time": 0,
                            "documents_found": 0,
                            "error": "Search failed"
                        }
                    }

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

            prompt = self.prompt_template.format(
                system_prompt=self.system_prompt,
                context=context,
                question=question
            )

            print(f"🔄 [DEBUG] Context length: {len(context)} characters")

            # Khởi tạo generation metrics
            generation_metrics = {
                "generation_time": 0,
                "generation_cost": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "successful_requests": 0,
                "errors": []
            }

            generation_start_time = time.time()

            try:
                # Sử dụng get_openai_callback để thu thập metrics
                with get_openai_callback() as cb:
                    response = await asyncio.wait_for(
                        self.llm.ainvoke([HumanMessage(content=prompt)]),
                        timeout=25.0
                    )

                    # Thu thập metrics từ callback
                    generation_metrics.update({
                        "generation_cost": cb.total_cost,
                        "prompt_tokens": cb.prompt_tokens,
                        "completion_tokens": cb.completion_tokens,
                        "total_tokens": cb.total_tokens,
                        "successful_requests": cb.successful_requests,
                    })

            except asyncio.TimeoutError:
                logger.warning("LLM response timeout")
                response = HumanMessage(content="Xin lỗi, hệ thống đang quá tải. Vui lòng thử lại sau.")
                generation_metrics["errors"].append("LLM timeout")
            except Exception as e:
                logger.error(f"Error in generation: {e}")
                response = HumanMessage(content="Xin lỗi, có lỗi xảy ra trong quá trình tạo câu trả lời.")
                generation_metrics["errors"].append(f"Generation error: {str(e)}")

            generation_end_time = time.time()
            generation_metrics["generation_time"] = generation_end_time - generation_start_time

            print(f"🤖 Generation completed:")
            print(f"   💰 Cost: ${generation_metrics['generation_cost']:.6f}")
            print(f"   ⏱️  Time: {generation_metrics['generation_time']:.3f}s")
            print(f"   📊 Tokens - Input: {generation_metrics['prompt_tokens']}, Output: {generation_metrics['completion_tokens']}, Total: {generation_metrics['total_tokens']}")

            # Combine vector metrics và generation metrics
            vector_metrics = state.get("vector_metrics", {})
            combined_metrics = {**vector_metrics, **generation_metrics}

            # Thu thập thông tin về retrieved documents
            retrieved_docs_info = []
            if docs:
                for i, doc in enumerate(docs):
                    doc_info = {
                        "chunk_index": i + 1,
                        "content": doc.page_content,
                        "content_length": len(doc.page_content),
                        "metadata": doc.metadata
                    }
                    retrieved_docs_info.append(doc_info)

            print(f"📄 Retrieved {len(retrieved_docs_info)} documents for generation")

            return {
                "messages": state["messages"] + [response],
                "docs": None,  # Clear docs after generation
                "vector_metrics": combined_metrics,  # ✅ Combine cả vector và generation metrics
                "retrieved_docs": retrieved_docs_info  # ✅ Thông tin về chunks được retrieve
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
            rewrite_result = {
                "can_process": True,
                "rewritten_query": query,
                "metrics": {
                    "pre_retrieval_time": 0,
                    "pre_retrieval_cost": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "subqueries_count": 1,
                    "errors": ["Pre-retrieval timeout"]
                }
            }

        processed_query = rewrite_result["rewritten_query"]
        pre_retrieval_metrics = rewrite_result.get("metrics", {})

        # Bước 2: Chuẩn bị state với subqueries
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "subqueries": rewrite_result.get("subqueries", [processed_query])
        }

        # Bước 3: Gọi LangGraph WITHOUT memory
        try:
            result = await asyncio.wait_for(
                self.graph_app.ainvoke(initial_state),
                timeout=40.0
            )
            final_answer = result["messages"][-1].content
            vector_metrics = result.get("vector_metrics", {})  # ✅ Lấy vector metrics từ LangGraph result
            retrieved_docs = result.get("retrieved_docs", [])  # ✅ Lấy retrieved docs từ LangGraph result
        except asyncio.TimeoutError:
            logger.warning("Graph execution timeout")
            final_answer = "Xin lỗi, hệ thống đang xử lý chậm. Vui lòng thử lại sau."
            vector_metrics = {}
            retrieved_docs = []

        return {
            "output": final_answer,
            "session_id": session_id,
            "subqueries": rewrite_result.get("subqueries", [processed_query]),
            "pre_retrieval_metrics": pre_retrieval_metrics,  # ✅ Từ pre-retrieval
            "vector_metrics": vector_metrics,  # ✅ Từ vector store (embedding + search timing)
            "retrieved_docs": retrieved_docs  # ✅ Chunks thực tế được retrieve từ RAG chat
        }
