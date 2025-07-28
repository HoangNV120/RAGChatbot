import asyncio
import json
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, AsyncGenerator
from app.master_chatbot import MasterChatbot
from app.document_processor import DocumentProcessor
from app.smart_scheduler import SmartDocumentScheduler
import uvicorn
from contextlib import asynccontextmanager
import os
import logging

# Cấu hình logging để hiển thị log routing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Global variables để store instances
master_chatbot = None
doc_processor = None
smart_scheduler = None

# Configuration for Google Drive integration
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "YOUR_FOLDER_ID_HERE")
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service-account.json")
SCHEDULER_INTERVAL_HOURS = int(os.getenv("SCHEDULER_INTERVAL_HOURS", "1"))  # Default: every hour

# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    global master_chatbot, doc_processor, smart_scheduler

    # Startup: Initialize components
    print("🚀 Initializing RAG Chatbot components...")

    # Create document processor
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "data")
    print(f"Setting data directory to: {data_dir}")
    doc_processor = DocumentProcessor(data_dir=data_dir)

    # Initialize Master Chatbot
    master_chatbot = MasterChatbot(vector_store=doc_processor.vector_store)

    # Initialize Smart Scheduler with Google Drive integration
    if GOOGLE_DRIVE_FOLDER_ID != "YOUR_FOLDER_ID_HERE":
        print("🔄 Initializing Smart Scheduler with Google Drive integration...")
        smart_scheduler = SmartDocumentScheduler(
            google_drive_folder_id=GOOGLE_DRIVE_FOLDER_ID,
            data_dir=data_dir,
            service_account_file=GOOGLE_SERVICE_ACCOUNT_FILE,
            schedule_interval_hours=SCHEDULER_INTERVAL_HOURS
        )

        # Start the scheduler
        smart_scheduler.start_scheduler()
        print(f"✅ Smart Scheduler started - will sync every {SCHEDULER_INTERVAL_HOURS} hour(s)")
    else:
        print("⚠️ Google Drive integration disabled - please set GOOGLE_DRIVE_FOLDER_ID")

        # Load documents manually if no Google Drive integration
        print("Loading documents manually...")
        await doc_processor.load_and_process_all_with_routing()
        print("✅ Documents loaded and indexed in the vector database")

    print("✅ RAG Chatbot is ready!")

    yield

    # Shutdown: Stop scheduler and cleanup
    print("🔄 Shutting down application...")
    if smart_scheduler:
        smart_scheduler.stop_scheduler()
    print("✅ Application shutdown complete")

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

@app.get("/health")
async def health_check():
    """
    Health check endpoint for production monitoring
    """
    try:
        # Check if master_chatbot is initialized
        if master_chatbot is None:
            return {"status": "unhealthy", "error": "Master chatbot not initialized"}

        # Check if doc_processor is initialized
        if doc_processor is None:
            return {"status": "unhealthy", "error": "Document processor not initialized"}

        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0",
            "services": {
                "master_chatbot": "running",
                "document_processor": "running",
                "scheduler": "running" if smart_scheduler else "stopped"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

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

@app.get("/chat/stream/{session_id}")
async def chat_stream_get(session_id: str, q: str):
    """
    SSE endpoint for streaming chat responses via GET request.
    Usage: GET /chat/stream/session123?q=your_question
    """
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")

    if len(q) > 255:
        raise HTTPException(status_code=400, detail="Query too long (max 255 characters)")

    async def event_generator():
        try:
            # Generate streaming response - chỉ stream chunk, done, error
            async for chunk in master_chatbot.generate_response_stream(
                query=q,
                session_id=session_id
            ):
                yield {
                    "event": "chunk",
                    "data": json.dumps(chunk)
                }

        except Exception as e:
            logger.error(f"Error in chat stream: {e}")
            yield {
                "event": "error",
                "data": json.dumps({
                    "type": "error",
                    "content": "🤖 Xin lỗi, có lỗi xảy ra. Bạn vui lòng thử lại sau.",
                    "timestamp": time.time()
                })
            }

    return EventSourceResponse(event_generator())

@app.get("/scheduler/status")
async def get_scheduler_status():
    """Get scheduler status and statistics"""
    if not smart_scheduler:
        return {"error": "Scheduler not initialized"}

    return smart_scheduler.get_status()

@app.post("/scheduler/manual-sync")
async def manual_sync():
    """Manually trigger sync and processing"""
    if not smart_scheduler:
        raise HTTPException(status_code=400, detail="Scheduler not initialized")

    try:
        result = await smart_scheduler.manual_sync()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Manual sync failed: {str(e)}")

@app.get("/vector-store/documents")
async def list_documents():
    """List all document names in vector store"""
    try:
        if not doc_processor:
            raise HTTPException(status_code=400, detail="Document processor not initialized")

        doc_names = await doc_processor.vector_store.list_all_document_names()
        return {"document_names": doc_names, "total_count": len(doc_names)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.delete("/vector-store/documents/{doc_name}")
async def delete_document(doc_name: str):
    """Delete a document from vector store by name"""
    try:
        if not doc_processor:
            raise HTTPException(status_code=400, detail="Document processor not initialized")

        success = await doc_processor.vector_store.delete_documents_by_name(doc_name)
        if success:
            return {"message": f"Successfully deleted document: {doc_name}"}
        else:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.get("/vector-store/documents/{doc_name}")
async def get_document(doc_name: str):
    """Get documents by name from vector store"""
    try:
        if not doc_processor:
            raise HTTPException(status_code=400, detail="Document processor not initialized")

        documents = await doc_processor.vector_store.get_documents_by_name(doc_name)
        return {
            "document_name": doc_name,
            "chunk_count": len(documents),
            "chunks": [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document: {str(e)}")

class DocumentUpdateRequest(BaseModel):
    doc_name: str
    content: str
    doc_type: str = "Manual"

@app.put("/vector-store/documents")
async def update_document(request: DocumentUpdateRequest):
    """Update a document in vector store"""
    try:
        if not doc_processor:
            raise HTTPException(status_code=400, detail="Document processor not initialized")

        from langchain.schema import Document

        # Create new document
        new_doc = Document(
            page_content=request.content,
            metadata={
                "name": request.doc_name,
                "type": request.doc_type
            }
        )

        # Split into chunks
        splits = await asyncio.get_event_loop().run_in_executor(
            None, lambda: doc_processor.text_splitter.split_documents([new_doc])
        )

        # Enhance with filename prefix
        enhanced_splits = []
        for split in splits:
            enhanced_content = f"{request.doc_name} : {split.page_content}"
            enhanced_split = Document(
                page_content=enhanced_content,
                metadata=split.metadata
            )
            enhanced_splits.append(enhanced_split)

        # Update in vector store
        success = await doc_processor.vector_store.update_documents_by_name(
            request.doc_name, enhanced_splits
        )

        if success:
            return {
                "message": f"Successfully updated document: {request.doc_name}",
                "chunk_count": len(enhanced_splits)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update document")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating document: {str(e)}")

if __name__ == "__main__":
    try:
        print("🔥 Starting RAG Chatbot Server...")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # Set to False for production
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
