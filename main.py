from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from app.master_chatbot import MasterChatbot
from app.document_processor import DocumentProcessor
import uvicorn
from contextlib import asynccontextmanager
import os

# Global variables để store instances
master_chatbot = None
doc_processor = None

# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    global master_chatbot, doc_processor

    # Startup: Initialize components
    print("🚀 Initializing RAG Chatbot components...")

    # Create document processor
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "data")
    print(f"Setting data directory to: {data_dir}")
    doc_processor = DocumentProcessor(data_dir=data_dir)

    # Initialize Master Chatbot
    master_chatbot = MasterChatbot(vector_store=doc_processor.vector_store)

    # Load documents into the vector database
    print("Loading documents into the vector database...")
    await doc_processor.load_and_process_all()
    print("✅ Documents loaded and indexed in the vector database")
    print("✅ RAG Chatbot is ready!")

    yield

    # Shutdown: Add any cleanup code here if needed
    print("🔄 Shutting down application...")

# Initialize FastAPI with lifespan
app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ChatRequest(BaseModel):
    chatInput: str
    sessionId: Optional[str] = None

class ChatResponse(BaseModel):
    output: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint to handle chat requests.
    Input format: {"chatInput": "pl13 cua k18", "sessionId": "abc"}
    Output format: {"output": "response text"}
    """
    if not request.chatInput:
        raise HTTPException(status_code=400, detail="Bạn chưa nhập câu hỏi. Vui lòng nhập câu hỏi để mình có thể hỗ trợ bạn nhé!")

    # Giới hạn độ dài chatInput tối đa 255 ký tự
    if len(request.chatInput) > 255:
        raise HTTPException(status_code=400, detail="Câu hỏi quá dài (tối đa 255 ký tự). Bạn vui lòng rút gọn lại để mình có thể hỗ trợ tốt hơn nhé!")

    # Generate response using the Master Chatbot
    response = await master_chatbot.generate_response(
        query=request.chatInput,
        session_id=request.sessionId
    )

    # Return just the output as specified in the format
    return ChatResponse(output=response["output"])

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
