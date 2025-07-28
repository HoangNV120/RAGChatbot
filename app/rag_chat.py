from typing import Dict, Optional, List, TypedDict, AsyncGenerator
from uuid import uuid4
import logging
import asyncio
import time
from functools import lru_cache

from langchain_core.messages import HumanMessage, BaseMessage
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_openai import ChatOpenAI
from app.vector_store import VectorStore
from app.pre_retrieval import PreRetrieval
from app.post_retrieval import PostRetrieval
from app.config import settings
import aiosqlite

# Cấu hình logging với level cao hơn để giảm overhead
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Khai báo state cho LangGraph
class GraphState(TypedDict):
    messages: List[BaseMessage]
    docs: Optional[List]
    subqueries: Optional[List[str]]  # Thêm field cho subqueries (câu hỏi phụ)

class RAGChat:
    _db_connection = None  # Shared connection pool
    _llm_instance = None   # Singleton LLM instance

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store if vector_store else VectorStore()
        self.query_rewriter = PreRetrieval()
        self.post_retrieval = PostRetrieval()  # Khởi tạo PostRetrieval để áp dụng reranking

        self.system_prompt = """Bạn là *Trợ lý Sinh viên FPTU* được huấn luyện để chỉ trả lời dựa trên thông tin có sẵn.
**NGUYÊN TẮC TUYỆT ĐỐI:**
- CHỈ sử dụng thông tin có trong Context được cung cấp
- KHÔNG được thêm kiến thức từ bên ngoài context
- KHÔNG được trả lời user query tổng quát (toán học, khoa học, lập trình)

**QUY TẮC TRÍCH DẪN:**
- Nếu context có đủ thông tin → Trả lời chính xác dựa trên context
- Nếu context có thông tin một phần → Trả lời phần có + "Để biết thêm chi tiết, bạn liên hệ Phòng CTSV"
- Nếu context không có thông tin → "Mình chưa có dữ liệu, bạn vui lòng liên hệ Phòng CTSV nhé."

**XỬ LÝ CÂU HỎI ĐẶC BIỆT:**
- Câu hỏi Yes/No: Kiểm tra thông tin trong context, trả lời "Đúng" hoặc "Không đúng" + giải thích dựa trên context
- Câu hỏi so sánh/xác minh: So sánh thông tin trong câu hỏi với thông tin trong context, nếu sai thì đưa ra thông tin đúng từ context

**ĐƯỢC PHÉP SỬ DỤNG:**
- So sánh thông tin có trong context
- Tổng hợp thông tin từ nhiều phần của context
- Phân tích mối quan hệ giữa các thông tin trong context
- Rút ra kết luận logic dựa trên thông tin có sẵn trong context
**CÁCH TRẢ LỜI:**
- Dùng "bạn/mình", thân thiện
- Trích dẫn trực tiếp từ context
- Có thể tổng hợp và so sánh thông tin trong context
- Không đặt câu hỏi ngược lại
**TUYỆT ĐỐI KHÔNG:**
- Sử dụng kiến thức tổng quát không có trong context
- Thêm thông tin từ bên ngoài context
- Giải thích khái niệm không có trong context"""

        # Sử dụng singleton LLM để tránh tạo mới nhiều lần
        if RAGChat._llm_instance is None:
            RAGChat._llm_instance = ChatOpenAI(
                model=settings.model_name,
                temperature=settings.temperature,
                api_key=settings.openai_api_key,
                max_retries=2,  # Giảm retries để phản hồi nhanh hơn
                timeout=30,  # Timeout 30s thay vì mặc định 60s
                streaming=True,
            )
        self.llm = RAGChat._llm_instance

        # Cache prompt template
        self.prompt_template = PromptTemplate.from_template(
            """{system_prompt}

---

Context được cung cấp: 
{context}

---

User query cần trả lời: {question}

---

**HƯỚNG DẪN XỬ LÝ:**
1. Đọc kỹ Context trên
2. Kiểm tra xem Context có chứa thông tin để trả lời user query không
3. Nếu CÓ → Trả lời dựa hoàn toàn trên thông tin trong Context
4. Nếu KHÔNG → Trả lời "Mình chưa có dữ liệu, bạn vui lòng liên hệ Phòng CTSV nhé."

**LƯU Ý:** Tuyệt đối không được thêm thông tin từ bên ngoài Context.

Trả lời:"""
        )

        # Khởi tạo LangGraph với memory để lưu lịch sử theo thread_id
        self.memory = None
        self.graph_app = None

    @classmethod
    async def _get_shared_db_connection(cls):
        """Singleton database connection để tránh tạo nhiều connection"""
        if cls._db_connection is None:
            cls._db_connection = await aiosqlite.connect(
                "chat_sessions.db",
                check_same_thread=False
            )
            # Tối ưu hóa SQLite
            await cls._db_connection.execute("PRAGMA journal_mode=WAL")
            await cls._db_connection.execute("PRAGMA synchronous=NORMAL")
            await cls._db_connection.execute("PRAGMA cache_size=10000")
            await cls._db_connection.execute("PRAGMA temp_store=memory")
        return cls._db_connection

    async def _ensure_graph_ready(self):
        """Đảm bảo graph đã được khởi tạo"""
        if self.graph_app is None:
            self.graph_app = await self._build_graph()

    async def _get_memory(self):
        """Lazy initialization của memory checkpointer với shared connection"""
        if self.memory is None:
            conn = await self._get_shared_db_connection()
            self.memory = AsyncSqliteSaver(conn)
        return self.memory

    @lru_cache(maxsize=128)
    def _format_prompt_cached(self, system_prompt: str, context: str, question: str) -> str:
        """Cache prompt formatting để tránh format lại"""
        return self.prompt_template.format(
            system_prompt=system_prompt,
            context=context,
            question=question
        )

    async def _build_graph(self):
        builder = StateGraph(GraphState)

        # Node: truy xuất tài liệu với parallel processing, subqueries và reranking
        async def retrieve(state: GraphState):
            question = state["messages"][-1].content

            # Lấy subqueries từ state nếu có, nếu không thì chỉ dùng câu hỏi gốc
            subqueries = state.get('subqueries', [question])

            print(len(subqueries))

            # Tối ưu: giảm k cho mỗi query để tổng số docs không quá lớn
            k_per_query = max(1, min(3, 6 // (len(subqueries) - 1)))

            # Tìm kiếm song song với tất cả các câu hỏi phụ
            search_tasks = []
            for query in subqueries:
                task = asyncio.create_task(
                    self.vector_store.similarity_search(query, k=k_per_query)
                )
                search_tasks.append(task)

            try:
                # Chờ tất cả các tìm kiếm hoàn thành với timeout
                all_results = await asyncio.wait_for(
                    asyncio.gather(*search_tasks, return_exceptions=True),
                    timeout=20.0
                )

                # Kết hợp và loại bỏ trùng lặp
                combined_docs = []
                seen_content = set()

                for results in all_results:
                    if isinstance(results, list):  # Kiểm tra không phải exception
                        for doc in results:
                            # Sử dụng hash của content để kiểm tra trùng lặp
                            content_hash = hash(doc.page_content)
                            if content_hash not in seen_content:
                                seen_content.add(content_hash)
                                combined_docs.append(doc)

                # Áp dụng LLM-based reranking nếu có nhiều documents
                # if len(combined_docs) > 4:
                #     try:
                #         # Sử dụng method llm_rerank public thay vì _llm_rerank private
                #         reranked_docs = await asyncio.wait_for(
                #             self.post_retrieval.llm_rerank(question, combined_docs, top_k=4),
                #             timeout=15.0
                #         )
                #         logger.info(f"LLM-based reranked {len(combined_docs)} docs to top {len(reranked_docs)}")
                #         final_docs = reranked_docs
                #     except asyncio.TimeoutError:
                #         logger.warning("LLM-based reranking timeout, using original docs")
                #         final_docs = combined_docs[:4]
                #     except Exception as e:
                #         logger.warning(f"LLM-based reranking failed: {e}, using original docs")
                #         final_docs = combined_docs[:4]
                # else:
                final_docs = combined_docs

                return {**state, "docs": final_docs}

            except asyncio.TimeoutError:
                logger.warning(f"Vector search timeout for subqueries")
                # Fallback: tìm kiếm với câu hỏi gốc
                try:
                    docs = await asyncio.wait_for(
                        self.vector_store.similarity_search(question, k=4),
                        timeout=10.0
                    )
                    return {**state, "docs": docs}
                except:
                    return {**state, "docs": []}

        # Node: tạo câu trả lời từ LLM với optimization
        async def generate(state: GraphState):
            question = state["messages"][-1].content
            docs = state["docs"]

            # Sử dụng toàn bộ context từ documents (bỏ giới hạn)
            if docs:
                context_parts = []
                for doc in docs:
                    context_parts.append(doc.page_content)
                context = "\n\n".join(context_parts)
            else:
                context = "Không tìm thấy thông tin liên quan."

            # Sử dụng cached prompt formatting
            prompt = self._format_prompt_cached(
                self.system_prompt,
                context,
                question
            )

            # Chỉ log khi cần thiết (level WARNING)
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"Context length: {len(context)} characters")

            try:
                # Thêm timeout cho LLM call
                response = await asyncio.wait_for(
                    self.llm.ainvoke([HumanMessage(content=prompt)]),
                    timeout=25.0  # Timeout 25s
                )
            except asyncio.TimeoutError:
                logger.warning("LLM response timeout")
                response = HumanMessage(content="Xin lỗi, hệ thống đang quá tải. Vui lòng thử lại sau.")

            return {
                "messages": state["messages"] + [response],
                "docs": None  # Xóa docs để không lưu vào memory
            }

        # Xây đồ thị
        builder.add_node("retrieve", retrieve)
        builder.add_node("generate", generate)

        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)

        # Compile với MemorySaver
        memory = await self._get_memory()
        return builder.compile(
            checkpointer=memory,
            interrupt_before=[],
            interrupt_after=[]
        )

    async def generate_response(self, query: str, session_id: Optional[str] = None) -> Dict:
        await self._ensure_graph_ready()

        if not session_id or session_id.strip() == "":
            session_id = str(uuid4())

        # Bước 1: Phân tích và viết lại query với timeout
        try:
            rewrite_result = await asyncio.wait_for(
                self.query_rewriter.analyze_and_rewrite(query),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning("Query rewriter timeout, using original query")
            rewrite_result = {"can_process": True, "rewritten_query": query}
        #
        # # Bước 2: Kiểm tra có thể xử lý không
        # if not rewrite_result["can_process"]:
        #     return {
        #         "output": "Mình thấy câu hỏi của bạn có nhiều chủ đề khác nhau. Để hỗ trợ tốt hơn, bạn có thể chia thành các câu hỏi riêng biệt không? 😊",
        #         "session_id": session_id,
        #         "messages": [query, "Không thể xử lý query phức tạp"]
        #     }

        processed_query = rewrite_result["rewritten_query"]

        # Bước 3: Lấy lịch sử chat từ memory với optimization
        config = {"configurable": {"thread_id": session_id}}

        try:
            # Timeout cho việc lấy state
            current_state = await asyncio.wait_for(
                self.graph_app.aget_state(config),
                timeout=3.0
            )
            existing_messages = current_state.values.get("messages", []) if current_state.values else []
        except (asyncio.TimeoutError, Exception):
            existing_messages = []

        # Thêm message mới và giới hạn history
        all_messages = existing_messages + [HumanMessage(content=processed_query)]

        # Chỉ lấy 3 messages gần nhất thay vì 5 để giảm context length
        recent_messages = all_messages[-3:] if len(all_messages) > 3 else all_messages

        # Bước 4: Chuẩn bị state với subqueries
        initial_state = {
            "messages": recent_messages,
            "subqueries": rewrite_result.get("subqueries", [processed_query])
        }

        # Bước 5: Gọi LangGraph với timeout
        try:
            result = await asyncio.wait_for(
                self.graph_app.ainvoke(initial_state, config=config),
                timeout=40.0  # Tổng timeout 40s
            )
            final_answer = result["messages"][-1].content
        except asyncio.TimeoutError:
            logger.warning("Graph execution timeout")
            final_answer = "Xin lỗi, hệ thống đang xử lý chậm. Vui lòng thử lại sau."
            result = {"messages": recent_messages + [HumanMessage(content=final_answer)]}

        return {
            "output": final_answer,
            "session_id": session_id,
            "messages": [msg.content for msg in result["messages"]],
            "subqueries": rewrite_result.get("subqueries", [processed_query])  # Thêm thông tin debug về các câu hỏi phụ
        }

    async def generate_response_stream(self, query: str, session_id: Optional[str] = None) -> AsyncGenerator[Dict, None]:
        """
        Streaming version sử dụng LangGraph astream
        """
        await self._ensure_graph_ready()

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

            # Bước 2: Lấy lịch sử chat từ memory
            config = {"configurable": {"thread_id": session_id}}

            try:
                current_state = await asyncio.wait_for(
                    self.graph_app.aget_state(config),
                    timeout=3.0
                )
                existing_messages = current_state.values.get("messages", []) if current_state.values else []
            except (asyncio.TimeoutError, Exception):
                existing_messages = []

            # Thêm message mới và giới hạn history
            all_messages = existing_messages + [HumanMessage(content=processed_query)]
            recent_messages = all_messages[-3:] if len(all_messages) > 3 else all_messages

            # Bước 3: Chuẩn bị state với subqueries
            initial_state = {
                "messages": recent_messages,
                "subqueries": rewrite_result.get("subqueries", [processed_query])
            }

            # Bước 4: Sử dụng LangGraph astream
            full_response = ""
            async for chunk in self.graph_app.astream(initial_state, config=config, stream_mode="updates"):
                # Xử lý chunk từ retrieve node
                if "retrieve" in chunk:
                    # Có thể yield thông tin về việc đang tìm kiếm
                    continue

                # Xử lý chunk từ generate node
                if "generate" in chunk:
                    state = chunk["generate"]
                    if "messages" in state and state["messages"]:
                        last_message = state["messages"][-1]
                        if hasattr(last_message, 'content'):
                            # Nếu là streaming response từ LLM
                            if hasattr(last_message, 'response_metadata') and last_message.response_metadata.get('streaming', False):
                                full_response += last_message.content
                                yield {
                                    "type": "chunk",
                                    "content": last_message.content,
                                    "timestamp": time.time()
                                }
                            else:
                                # Nếu là complete response, stream theo từng chunk
                                response_content = last_message.content
                                if response_content != full_response:
                                    new_content = response_content[len(full_response):]
                                    full_response = response_content

                                    # Stream từng từ để có hiệu ứng typing
                                    words = new_content.split()
                                    for word in words:
                                        yield {
                                            "type": "chunk",
                                            "content": word + " ",
                                            "timestamp": time.time()
                                        }
                                        await asyncio.sleep(0.01)  # Small delay

            # Yield done event
            yield {
                "type": "done",
                "session_id": session_id,
                "route_used": "RAG_CHAT",
                "routing_info": {},
                "timestamp": time.time(),
                "subqueries": rewrite_result.get("subqueries", [processed_query]),
                "final_answer": full_response
            }

        except Exception as e:
            logger.error(f"Error in RAG streaming: {e}")
            yield {
                "type": "error",
                "content": "🤖 Xin lỗi, có lỗi xảy ra. Bạn vui lòng thử lại sau.",
                "timestamp": time.time()
            }

