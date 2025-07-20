# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from app.master_chatbot import MasterChatbot
from app.document_processor import DocumentProcessor
from app.vector_store_small import VectorStoreSmall
import uvicorn
from contextlib import asynccontextmanager
import os
import logging
import pandas as pd
from langchain.schema import Document

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
vector_store_small = None

async def load_data_to_ragsmall(vector_store_small):
    """
    Load Excel data vào ragsmall collection - chỉ embed question
    """
    try:
        # Check if collection already has data
        if vector_store_small.collection_exists_and_has_data():
            info = vector_store_small.get_collection_info()
            print(f"[INFO] ragsmall collection already exists with {info.get('points_count', 0)} documents")
            print(f"[INFO] Skipping data loading to avoid duplicates")
            return True
        
        # Tìm file Excel
        excel_path = os.path.join("app", "data_test.xlsx")
        
        if not os.path.exists(excel_path):
            print(f"[ERROR] Excel file not found at: {excel_path}")
            return False
        
        # Load Excel file
        df = pd.read_excel(excel_path)
        print(f"[INFO] Loaded {len(df)} rows from Excel for ragsmall")
        
        if 'question' not in df.columns or 'answer' not in df.columns:
            print("[ERROR] Excel file must contain 'question' and 'answer' columns")
            return False
        
        # Process documents - CHỈ EMBED QUESTION
        documents = []
        
        for idx, row in df.iterrows():
            try:
                questions = str(row['question']).split('|')
                answer = str(row['answer'])
                
                # Extract metadata
                category = "general"
                if 'category' in df.columns:
                    category = str(row['category']).strip()
                    if category.lower() in ['nan', 'none', '']:
                        category = "general"
                
                source = "unknown"
                if 'nguồn' in df.columns:
                    source = str(row['nguồn']).strip()
                    if source.lower() in ['nan', 'none', '']:
                        source = "unknown"
                
                # Tạo document cho mỗi question - CHỈ EMBED QUESTION
                for q in questions:
                    q = q.strip()
                    if q:
                        # Page content CHỈ LÀ QUESTION (không có "Question:" prefix)
                        doc = Document(
                            page_content=q,  # CHỈ question thôi
                            metadata={
                                "question": q,
                                "answer": answer,
                                "category": category,
                                "source": source,
                                "type": "FQA_SMALL"
                            }
                        )
                        documents.append(doc)
                        
            except Exception as e:
                print(f"[ERROR] Error processing row {idx}: {e}")
                continue
        
        if not documents:
            print("[ERROR] No valid documents created for ragsmall")
            return False
        
        print(f"[INFO] Created {len(documents)} documents for ragsmall (question-only format)")
        
        # Add to vector store
        await vector_store_small.add_documents(documents)
        print(f"[SUCCESS] Added {len(documents)} documents to ragsmall collection")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error loading data to ragsmall: {e}")
        return False

# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    global master_chatbot, doc_processor, vector_store_small

    # Startup: Initialize components
    print("[STARTUP] Initializing Multi-Collection RAG Chatbot components...")
    print("=" * 60)

    # 1. Khởi tạo Document processor và vector store cho RAG (ragchatbot collection)
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "data")
    print(f"[INFO] Setting data directory to: {data_dir}")
    doc_processor = DocumentProcessor(data_dir=data_dir)

    # 2. Khởi tạo VectorStoreSmall cho ragsmall collection
    print(f"[INFO] Initializing ragsmall vector store...")
    vector_store_small = VectorStoreSmall()

    # 3. Initialize Master Chatbot với dual vector stores
    print(f"[INFO] Initializing Master Chatbot with dual vector stores...")
    master_chatbot = MasterChatbot(
        vector_store=doc_processor.vector_store,  # Main vector store (ragchatbot)
        vector_store_small=vector_store_small     # Small vector store (ragsmall)
    )

    # 4. Load documents vào cả hai collections
    print(f"\n[INFO] Loading documents into vector databases...")
    
    # Load vào ragchatbot collection (format: "Question: ... Answer: ...")
    print(f"[INFO] Loading data to ragchatbot collection...")
    await doc_processor.load_and_process_excel()
    print(f"[SUCCESS] ragchatbot collection loaded!")
    
    # Load vào ragsmall collection (chỉ question)
    print(f"[INFO] Loading data to ragsmall collection...")
    success = await load_data_to_ragsmall(vector_store_small)
    if success:
        print(f"[SUCCESS] ragsmall collection loaded!")
    else:
        print(f"[ERROR] Failed to load ragsmall collection!")
    
    # 5. Show collection info
    print(f"\n[INFO] Collection Statistics:")
    try:
        # ragchatbot info
        main_info = doc_processor.vector_store.client.get_collection(doc_processor.vector_store.vector_store.collection_name)
        print(f"   [INFO] ragchatbot: {main_info.points_count} documents")
    except Exception as e:
        print(f"   [ERROR] ragchatbot error: {e}")
    
    # ragsmall info
    small_info = vector_store_small.get_collection_info()
    if 'error' not in small_info:
        print(f"   [INFO] ragsmall: {small_info.get('points_count', 'unknown')} documents")
    else:
        print(f"   [ERROR] ragsmall error: {small_info['error']}")
    
    print(f"\n[SUCCESS] Multi-Collection RAG Chatbot is ready!")
    print("=" * 60)

    yield

    # Shutdown: Add any cleanup code here if needed
    print("[SHUTDOWN] Shutting down application...")

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

    # Generate response using the Master Chatbot với dual vector store flow
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

@app.get("/collections/info")
async def get_collections_info():
    """Get information about both collections"""
    try:
        info = {}
        
        # ragchatbot collection info
        try:
            main_info = doc_processor.vector_store.client.get_collection(doc_processor.vector_store.vector_store.collection_name)
            info["ragchatbot"] = {
                "points_count": main_info.points_count,
                "vectors_count": main_info.vectors_count,
                "status": main_info.status,
                "collection_name": doc_processor.vector_store.vector_store.collection_name
            }
        except Exception as e:
            info["ragchatbot"] = {"error": str(e)}
        
        # ragsmall collection info
        small_info = vector_store_small.get_collection_info()
        info["ragsmall"] = small_info
        
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collections info: {str(e)}")

@app.post("/chat/compare")
async def compare_collections_search(request: ChatRequest):
    """Compare search results between ragchatbot and ragsmall collections"""
    if not request.chatInput:
        raise HTTPException(status_code=400, detail="Query is required")
    
    try:
        results = {}
        
        # Search in ragchatbot (main collection)
        try:
            main_results = await doc_processor.vector_store.similarity_search_with_score(
                query=request.chatInput,
                k=3
            )
            
            results["ragchatbot"] = [
                {
                    "score": float(score),
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "category": doc.metadata.get('category', 'NO_CATEGORY'),
                    "source": doc.metadata.get('source', 'NO_SOURCE')
                }
                for doc, score in main_results
            ]
        except Exception as e:
            results["ragchatbot"] = {"error": str(e)}
        
        # Search in ragsmall (now integrated in master_chatbot)
        try:
            small_results = await vector_store_small.similarity_search_with_score(
                query=request.chatInput,
                k=3
            )
            
            results["ragsmall"] = [
                {
                    "score": float(score),
                    "question": doc.page_content,
                    "answer": doc.metadata.get('answer', '')[:200] + "..." if len(doc.metadata.get('answer', '')) > 200 else doc.metadata.get('answer', ''),
                    "category": doc.metadata.get('category', 'NO_CATEGORY'),
                    "source": doc.metadata.get('source', 'NO_SOURCE')
                }
                for doc, score in small_results
            ]
        except Exception as e:
            results["ragsmall"] = {"error": str(e)}
        
        return {
            "query": request.chatInput,
            "results": results,
            "note": "Comparison between dual vector stores used by single MasterChatbot"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing collections: {str(e)}")

@app.post("/router/test")
async def test_category_router(request: ChatRequest):
    """Test CategoryPartitionedRouter với dual vector store flow"""
    if not request.chatInput:
        raise HTTPException(status_code=400, detail="Query is required")
    
    try:
        # Test classification only (không full routing)
        classification_result = await master_chatbot.router._classify_query_category(request.chatInput)
        
        return {
            "query": request.chatInput,
            "classification_result": classification_result,
            "note": "New flow: Single chatbot with dual vector stores"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing router: {str(e)}")

@app.get("/router/stats")
async def get_router_stats():
    """Get router statistics for dual vector store setup"""
    try:
        stats = {
            "architecture": "Single MasterChatbot with dual vector stores",
            "main_vector_store": "ragchatbot (for RAG_CHAT)",
            "small_vector_store": "ragsmall (for quick search)",
            "router_stats": master_chatbot.router.get_stats()
        }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting router stats: {str(e)}")

if __name__ == "__main__":
    print("[STARTUP] Starting RAG Chatbot server...")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        log_level="info"
    )
