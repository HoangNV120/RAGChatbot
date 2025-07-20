from typing import Dict, Optional
import logging

from app.category_partitioned_router import CategoryPartitionedRouter
from app.rag_chat import RAGChat
# from app.rule_based_chatbot import AdvancedChatbot  # TẮT TẠM THỜI
from app.safety_guard import SafetyGuard
from app.config import settings

# Loại bỏ basicConfig để tránh xung đột với main.py
logger = logging.getLogger(__name__)

class MasterChatbot:
    def __init__(self, vector_store=None, vector_store_small=None):
        """
        Khởi tạo Master Chatbot với dual vector stores
        Args:
            vector_store: Vector store chính (ragchatbot) cho RAG_CHAT
            vector_store_small: Vector store nhỏ (ragsmall) cho quick search
        """
        # Khởi tạo safety guard - kiểm tra đầu tiên
        self.safety_guard = SafetyGuard()

        # Store both vector stores
        self.vector_store = vector_store  # ragchatbot collection
        self.vector_store_small = vector_store_small  # ragsmall collection
        
        if not self.vector_store:
            raise ValueError("Main vector_store is required for MasterChatbot")
        if not self.vector_store_small:
            raise ValueError("vector_store_small is required for the new dual-flow architecture")

        # Khởi tạo category-partitioned router với vector_store_small cho quick search
        self.router = CategoryPartitionedRouter(
            vector_store=vector_store_small,  # Use ragsmall for quick category search
            use_categorized_data=True
        )

        # RAG Chat sử dụng vector_store chính cho full processing
        self.rag_chat = RAGChat(vector_store=vector_store)

        # TẮT TẠM THỜI Rule-based chatbot
        # self.rule_based_chatbot = AdvancedChatbot()
        
        print(f"[INFO] MasterChatbot initialized with dual vector stores:")
        print(f"  - Main vector_store: {type(vector_store).__name__} (for RAG_CHAT)")
        print(f"  - Small vector_store: {type(vector_store_small).__name__} (for quick search)")

    async def generate_response(self, query: str, session_id: Optional[str] = None) -> Dict:
        """
        Phương thức chính để xử lý query với dual vector store flow
        
        Luồng mới:
        1. LLM Classification
        2. If category = "KHÁC" → Direct RAG_CHAT (main vector_store)
        3. If category hợp lệ → ragsmall search (vector_store_small) 
        4. If ragsmall similarity ≥ 0.8 → Return ragsmall answer
        5. If ragsmall similarity < 0.8 → Fallback to RAG_CHAT (main vector_store)
        """
        try:
            print(f"\n[PROCESSING] Query: {query[:50]}...")
            
            # Bước 1: LLM Classification only (không dùng full routing)
            classification_result = await self.router._classify_query_category(query)
            category = classification_result["category"] 
            should_use_vector = classification_result["should_use_vector"]
            
            print(f"[CLASSIFICATION] Category: {category}")
            print(f"[CLASSIFICATION] Should use vector: {should_use_vector}")
            
            # Bước 2: Route Decision theo luồng mới
            if not should_use_vector or category == "KHÁC":
                print(f"[ROUTE] Category '{category}' → Direct RAG_CHAT (main vector_store)")
                
                # Direct RAG_CHAT với main vector_store
                result = await self.rag_chat.generate_response(query, session_id)
                result["route_used"] = "RAG_CHAT_DIRECT"
                result["classification"] = classification_result
                return result
            
            # Bước 3: ragsmall Quick Search (vector_store_small)
            print(f"[ROUTE] Category '{category}' → ragsmall search (vector_store_small)")
            
            try:
                # Search trong ragsmall với k=5 như yêu cầu
                ragsmall_results = await self.vector_store_small.similarity_search_with_score(
                    query=query,
                    k=5
                )
                
                if not ragsmall_results:
                    print(f"[RAGSMALL] No results found → Fallback to RAG_CHAT")
                    result = await self.rag_chat.generate_response(query, session_id)
                    result["route_used"] = "RAG_CHAT_FALLBACK"
                    result["classification"] = classification_result
                    result["ragsmall_reason"] = "No results found"
                    return result
                
                # Filter by category và lấy best match
                category_results = []
                found_categories = {}
                
                for doc, similarity in ragsmall_results:
                    doc_category = doc.metadata.get('category', '').strip().upper()
                    
                    # Track categories
                    if doc_category in found_categories:
                        found_categories[doc_category] += 1
                    else:
                        found_categories[doc_category] = 1
                    
                    # Category matching (exact hoặc fuzzy)
                    if doc_category == category or \
                       doc_category.replace(' ', '') == category.replace(' ', '') or \
                       (doc_category and category and doc_category in category) or \
                       (doc_category and category and category in doc_category):
                        category_results.append((doc, similarity))
                
                print(f"[RAGSMALL] Found {len(ragsmall_results)} total, {len(category_results)} in category '{category}'")
                print(f"[RAGSMALL] Categories found: {found_categories}")
                
                # Nếu không có results trong category, thử fuzzy matching
                if not category_results:
                    for available_cat in found_categories.keys():
                        if available_cat and category:
                            if category.replace(' ', '').lower() in available_cat.replace(' ', '').lower() or \
                               available_cat.replace(' ', '').lower() in category.replace(' ', '').lower():
                                print(f"[RAGSMALL] Trying fuzzy match with '{available_cat}'")
                                for doc, similarity in ragsmall_results:
                                    if doc.metadata.get('category', '').strip().upper() == available_cat:
                                        category_results.append((doc, similarity))
                                break
                
                # Bước 4: Check threshold với best result
                if category_results:
                    # Sort by similarity và lấy best
                    category_results.sort(key=lambda x: x[1], reverse=True)
                    best_doc, best_similarity = category_results[0]
                    
                    print(f"[RAGSMALL] Best similarity: {best_similarity:.3f}")
                    print(f"[RAGSMALL] Threshold check: {best_similarity:.3f} >= 0.8?")
                    
                    if best_similarity >= 0.8:
                        # Return answer từ ragsmall
                        print(f"[SUCCESS] ragsmall match found (similarity: {best_similarity:.3f})")
                        
                        answer = best_doc.metadata.get('answer', '')
                        matched_question = best_doc.page_content
                        source = best_doc.metadata.get('source', '')
                        
                        return {
                            "output": answer,
                            "session_id": session_id or "ragsmall-session",
                            "route_used": "RAGSMALL_MATCH",
                            "classification": classification_result,
                            "ragsmall_info": {
                                "similarity_score": best_similarity,
                                "matched_question": matched_question,
                                "matched_category": category,
                                "source": source,
                                "total_results": len(ragsmall_results),
                                "category_results": len(category_results)
                            }
                        }
                
                # Bước 5: Fallback to RAG_CHAT nếu similarity < 0.8 hoặc không có category match
                print(f"[FALLBACK] ragsmall similarity < 0.8 or no category match → RAG_CHAT")
                result = await self.rag_chat.generate_response(query, session_id)
                result["route_used"] = "RAG_CHAT_FALLBACK"
                result["classification"] = classification_result
                result["ragsmall_info"] = {
                    "best_similarity": category_results[0][1] if category_results else 0.0,
                    "total_results": len(ragsmall_results),
                    "category_results": len(category_results),
                    "found_categories": found_categories
                }
                return result
                
            except Exception as e:
                print(f"[ERROR] ragsmall search failed: {e}")
                # Fallback to RAG_CHAT on error
                result = await self.rag_chat.generate_response(query, session_id)
                result["route_used"] = "RAG_CHAT_ERROR_FALLBACK"  
                result["classification"] = classification_result
                result["ragsmall_error"] = str(e)
                return result

        except Exception as e:
            logger.error(f"Error in master chatbot: {e}")
            return {
                "output": "🤖 Xin lỗi, có lỗi xảy ra. Bạn vui lòng thử lại sau.",
                "session_id": session_id or "error-session",
                "route_used": "ERROR",
                "error": str(e)
            }
