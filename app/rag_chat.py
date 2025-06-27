from typing import Dict, Optional, List, TypedDict
from uuid import uuid4
import logging

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from app.vector_store import VectorStore
from app.query_rewriter import QueryRewriter
from app.config import settings

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khai báo state cho LangGraph
class GraphState(TypedDict):
    messages: List[BaseMessage]
    docs: Optional[List]

class RAGChat:
    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store if vector_store else VectorStore()
        self.query_rewriter = QueryRewriter()

        self.system_prompt = """Bạn là *Trợ lý Sinh viên FPTU*.
Mục tiêu: trả lời chính xác, đầy đủ, văn phong thân thiện-chuyên nghiệp.

Quy tắc:
1. Dùng đại từ "bạn / mình".
2. Nếu chưa chắc thông tin, chỉ nói "Mình chưa có dữ liệu, bạn liên hệ Phòng CTSV."
3. Không tiết lộ email nội bộ, dữ liệu riêng tư.
4. Luôn thêm nguồn (link FAP hoặc tài liệu nội bộ) nếu có.

Cấu trúc trả lời:
- **Tóm tắt**: 1–2 câu trả lời trực tiếp.
- **Chi tiết**: Dẫn chính xác từ tài liệu.
- **Nguồn**: Tên tài liệu hoặc link.

Chỉ trả lời nếu context hỗ trợ đầy đủ và rõ ràng."""

        self.llm = ChatOpenAI(
            model=settings.model_name,
            temperature=settings.temperature,
            api_key=settings.openai_api_key
        )

        self.prompt_template = PromptTemplate.from_template(
            """
{system_prompt}

Thông tin truy xuất:
---------------------
{context}
---------------------

Câu hỏi: {question}

Hãy trả lời câu hỏi dựa trên thông tin trong context. Nếu context có thông tin liên quan, hãy sử dụng nó để trả lời. Chỉ trả lời "Mình chưa có dữ liệu, bạn liên hệ Phòng CTSV." khi context hoàn toàn không có thông tin liên quan đến câu hỏi.
"""
        )

        # Khởi tạo LangGraph với MemorySaver để lưu lịch sử theo thread_id
        self.memory = MemorySaver()
        self.graph_app = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(GraphState)

        # Node: truy xuất tài liệu
        async def retrieve(state: GraphState):
            question = state["messages"][-1].content
            docs = await self.vector_store.similarity_search(question, k=4)
            logger.info(f"Retrieved {len(docs)} docs.")
            return {**state, "docs": docs}

        # Node: tạo câu trả lời từ LLM
        async def generate(state: GraphState):
            question = state["messages"][-1].content
            docs = state["docs"]
            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = self.prompt_template.format(
                system_prompt=self.system_prompt,
                context=context,
                question=question
            )

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return {
                **state,
                "messages": state["messages"] + [response]
            }

        # Node: cập nhật message history (nếu muốn xử lý gì thêm)
        async def update_memory(state: GraphState):
            return state

        # Xây đồ thị
        builder.add_node("retrieve", retrieve)
        builder.add_node("generate", generate)
        builder.add_node("update_memory", update_memory)

        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", "update_memory")
        builder.add_edge("update_memory", END)

        # Compile với MemorySaver
        return builder.compile(checkpointer=self.memory)

    async def generate_response(self, query: str, session_id: Optional[str] = None) -> Dict:
        if not session_id:
            session_id = str(uuid4())

        # Bước 1: Phân tích và viết lại query
        rewrite_result = await self.query_rewriter.analyze_and_rewrite(query)

        # Bước 2: Kiểm tra có thể xử lý không
        if not rewrite_result["can_process"]:
            return {
                "output": "Mình thấy câu hỏi của bạn có nhiều chủ đề khác nhau. Để hỗ trợ tốt hơn, bạn có thể chia thành các câu hỏi riêng biệt không? 😊",
                "session_id": session_id,
                "messages": [query, "Không thể xử lý query phức tạp"]
            }

        # Bước 3: Sử dụng query đã được viết lại để tìm kiếm
        processed_query = rewrite_result["rewritten_query"]
        logger.info(f"Original query: {query}")
        logger.info(f"Rewritten query: {processed_query}")

        # Bước 4: Gọi LangGraph với query đã được cải thiện
        result = await self.graph_app.ainvoke(
            {"messages": [HumanMessage(content=processed_query)]},
            config={"configurable": {"thread_id": session_id}}
        )

        final_answer = result["messages"][-1].content
        return {
            "output": final_answer,
            "session_id": session_id,
            "messages": [msg.content for msg in result["messages"]]
        }






