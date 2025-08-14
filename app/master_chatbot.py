from typing import Dict, Optional
import logging
import time
import asyncio

from app.category_partitioned_router import CategoryPartitionedRouter
from app.rag_chat import RAGChat
from app.rag_chat_streaming import RAGChatStreaming
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

        # Store both vector stores
        self.vector_store = vector_store  # ragchatbot collection
        self.vector_store_small = vector_store_small  # ragsmall collection

        if not self.vector_store:
            raise ValueError("Main vector_store is required for MasterChatbot")
        if not self.vector_store_small:
            raise ValueError("vector_store_small is required for the new dual-flow architecture")

        # RAG Chat (no streaming, no memory) sử dụng vector_store chính
        self.rag_chat = RAGChat(vector_store=vector_store)

        # RAG Chat Streaming (with memory) sử dụng vector_store chính
        self.rag_chat_streaming = RAGChatStreaming(vector_store=vector_store)

        print(f"[INFO] MasterChatbot initialized with dual vector stores and dual RAG modes:")
        print(f"  - Main vector_store: {type(vector_store).__name__} (for RAG_CHAT)")
        print(f"  - Small vector_store: {type(vector_store_small).__name__} (for quick search)")

    async def generate_response(self, query: str, session_id: Optional[str] = None) -> Dict:
        """
        Phương thức chính để xử lý query với dual vector store flow

        Luồng mới (bỏ qua classification):
        1. Direct search trong ragsmall (vector_store_small)
        2. If ragsmall similarity ≥ 0.8 → Return ragsmall answer
        3. If ragsmall similarity < 0.8 → Fallback to RAG_CHAT (main vector_store)
        """
        try:
            # 🕐 START - Tổng thời gian processing
            total_start_time = time.time()
            print(f"\n🕐 [TIMER] TOTAL START: {time.strftime('%H:%M:%S', time.localtime(total_start_time))}")
            print(f"\n[PROCESSING] Query: {query[:50]}...")

            # 🕐 Bước 1: Direct ragsmall Quick Search (vector_store_small) - BỎ QUA CLASSIFICATION
            print(f"[ROUTE] Direct ragsmall search (vector_store_small) - Skip classification")

            try:
                # 🕐 Search trong ragsmall với k=5
                ragsmall_search_start_time = time.time()
                print(f"🕐 [TIMER] RAGSMALL_SEARCH START: {time.strftime('%H:%M:%S', time.localtime(ragsmall_search_start_time))}")

                ragsmall_results = await self.vector_store_small.similarity_search_with_score(
                    query=query,
                    k=5
                )

                ragsmall_search_end_time = time.time()
                ragsmall_search_duration = ragsmall_search_end_time - ragsmall_search_start_time
                print(f"🕐 [TIMER] RAGSMALL_SEARCH END: {time.strftime('%H:%M:%S', time.localtime(ragsmall_search_end_time))} - Duration: {ragsmall_search_duration:.3f}s")

                if not ragsmall_results:
                    print(f"[RAGSMALL] No results found → Fallback to RAG_CHAT")

                    # 🕐 Fallback RAG_CHAT
                    fallback_rag_start_time = time.time()
                    print(f"🕐 [TIMER] RAG_CHAT_FALLBACK START: {time.strftime('%H:%M:%S', time.localtime(fallback_rag_start_time))}")

                    result = await self.rag_chat.generate_response(query, session_id)

                    fallback_rag_end_time = time.time()
                    fallback_rag_duration = fallback_rag_end_time - fallback_rag_start_time
                    print(f"🕐 [TIMER] RAG_CHAT_FALLBACK END: {time.strftime('%H:%M:%S', time.localtime(fallback_rag_end_time))} - Duration: {fallback_rag_duration:.3f}s")

                    result["route_used"] = "RAG_CHAT_FALLBACK"
                    result["ragsmall_reason"] = "No results found"

                    # 🕐 TOTAL END
                    total_end_time = time.time()
                    total_duration = total_end_time - total_start_time
                    print(f"🕐 [TIMER] TOTAL END: {time.strftime('%H:%M:%S', time.localtime(total_end_time))} - Total Duration: {total_duration:.3f}s")
                    print(f"🕐 [BREAKDOWN] Ragsmall Search: {ragsmall_search_duration:.3f}s | RAG_CHAT: {fallback_rag_duration:.3f}s")

                    return result

                # 🕐 Bước 2: Check threshold với best result
                threshold_check_start_time = time.time()
                print(f"🕐 [TIMER] THRESHOLD_CHECK START: {time.strftime('%H:%M:%S', time.localtime(threshold_check_start_time))}")

                # Sort by similarity và lấy best
                ragsmall_results.sort(key=lambda x: x[1], reverse=True)
                best_doc, best_similarity = ragsmall_results[0]

                print(f"[RAGSMALL] Best similarity: {best_similarity:.3f}")
                print(f"[RAGSMALL] Threshold check: {best_similarity:.3f} >= 0.8?")

                if best_similarity >= 0.8:
                    # Return answer từ ragsmall
                    print(f"[SUCCESS] ragsmall match found (similarity: {best_similarity:.3f})")

                    answer = best_doc.metadata.get('answer', '')
                    matched_question = best_doc.page_content
                    source = best_doc.metadata.get('source', '')
                    category = best_doc.metadata.get('category', '')

                    threshold_check_end_time = time.time()
                    threshold_check_duration = threshold_check_end_time - threshold_check_start_time
                    print(f"🕐 [TIMER] THRESHOLD_CHECK END: {time.strftime('%H:%M:%S', time.localtime(threshold_check_end_time))} - Duration: {threshold_check_duration:.3f}s")

                    # 🕐 TOTAL END - SUCCESS
                    total_end_time = time.time()
                    total_duration = total_end_time - total_start_time
                    print(f"🕐 [TIMER] TOTAL END: {time.strftime('%H:%M:%S', time.localtime(total_end_time))} - Total Duration: {total_duration:.3f}s")
                    print(f"🕐 [BREAKDOWN] Ragsmall Search: {ragsmall_search_duration:.3f}s | Threshold Check: {threshold_check_duration:.3f}s")

                    return {
                        "output": answer,
                        "session_id": session_id or "ragsmall-session",
                        "route_used": "RAGSMALL_MATCH",
                        "ragsmall_info": {
                            "similarity_score": best_similarity,
                            "matched_question": matched_question,
                            "matched_category": category,
                            "source": source,
                            "total_results": len(ragsmall_results)
                        },
                        "timing_info": {
                            "total_duration": total_duration,
                            "ragsmall_search_duration": ragsmall_search_duration,
                            "threshold_check_duration": threshold_check_duration
                        }
                    }

                threshold_check_end_time = time.time()
                threshold_check_duration = threshold_check_end_time - threshold_check_start_time
                print(f"🕐 [TIMER] THRESHOLD_CHECK END: {time.strftime('%H:%M:%S', time.localtime(threshold_check_end_time))} - Duration: {threshold_check_duration:.3f}s")

                # 🕐 Bước 3: Fallback to RAG_CHAT nếu similarity < 0.8
                print(f"[FALLBACK] ragsmall similarity < 0.8 ({best_similarity:.3f}) → RAG_CHAT")

                fallback_rag_start_time = time.time()
                print(f"🕐 [TIMER] RAG_CHAT_FALLBACK START: {time.strftime('%H:%M:%S', time.localtime(fallback_rag_start_time))}")

                result = await self.rag_chat.generate_response(query, session_id)

                fallback_rag_end_time = time.time()
                fallback_rag_duration = fallback_rag_end_time - fallback_rag_start_time
                print(f"🕐 [TIMER] RAG_CHAT_FALLBACK END: {time.strftime('%H:%M:%S', time.localtime(fallback_rag_end_time))} - Duration: {fallback_rag_duration:.3f}s")

                result["route_used"] = "RAG_CHAT_FALLBACK"
                result["ragsmall_info"] = {
                    "best_similarity": best_similarity,
                    "total_results": len(ragsmall_results),
                    "best_category": best_doc.metadata.get('category', ''),
                    "threshold_reason": f"Similarity {best_similarity:.3f} < 0.8"
                }

                # 🕐 TOTAL END
                total_end_time = time.time()
                total_duration = total_end_time - total_start_time
                print(f"🕐 [TIMER] TOTAL END: {time.strftime('%H:%M:%S', time.localtime(total_end_time))} - Total Duration: {total_duration:.3f}s")
                print(f"🕐 [BREAKDOWN] Ragsmall Search: {ragsmall_search_duration:.3f}s | Threshold Check: {threshold_check_duration:.3f}s | RAG_CHAT: {fallback_rag_duration:.3f}s")

                return result

            except Exception as e:
                print(f"[ERROR] ragsmall search failed: {e}")

                # 🕐 Fallback to RAG_CHAT on error
                error_fallback_start_time = time.time()
                print(f"🕐 [TIMER] RAG_CHAT_ERROR_FALLBACK START: {time.strftime('%H:%M:%S', time.localtime(error_fallback_start_time))}")

                result = await self.rag_chat.generate_response(query, session_id)

                error_fallback_end_time = time.time()
                error_fallback_duration = error_fallback_end_time - error_fallback_start_time
                print(f"🕐 [TIMER] RAG_CHAT_ERROR_FALLBACK END: {time.strftime('%H:%M:%S', time.localtime(error_fallback_end_time))} - Duration: {error_fallback_duration:.3f}s")

                result["route_used"] = "RAG_CHAT_ERROR_FALLBACK"
                result["ragsmall_error"] = str(e)

                # 🕐 TOTAL END
                total_end_time = time.time()
                total_duration = total_end_time - total_start_time
                print(f"🕐 [TIMER] TOTAL END: {time.strftime('%H:%M:%S', time.localtime(total_end_time))} - Total Duration: {total_duration:.3f}s")
                print(f"🕐 [BREAKDOWN] RAG_CHAT (Error): {error_fallback_duration:.3f}s")

                return result

        except Exception as e:
            logger.error(f"Error in master chatbot: {e}")
            return {
                "output": "🤖 Xin lỗi, có lỗi xảy ra. Bạn vui lòng thử lại sau.",
                "session_id": session_id or "error-session",
                "route_used": "ERROR",
                "error": str(e)
            }

    async def generate_response_stream(self, query: str, session_id: Optional[str] = None):
        """
        Phương thức streaming để xử lý query với dual vector store flow

        Luồng mới (bỏ qua classification):
        1. Direct search trong ragsmall (vector_store_small)
        2. If ragsmall similarity ≥ 0.8 → Stream ragsmall answer
        3. If ragsmall similarity < 0.8 → Fallback to RAG_CHAT stream
        """
        try:
            # 🕐 START - Tổng thời gian processing
            total_start_time = time.time()
            print(f"\n🕐 [STREAM TIMER] TOTAL START: {time.strftime('%H:%M:%S', time.localtime(total_start_time))}")
            print(f"\n[STREAM PROCESSING] Query: {query[:50]}...")

            # 🕐 Bước 1: Direct ragsmall Quick Search (vector_store_small) - BỎ QUA CLASSIFICATION
            print(f"[STREAM ROUTE] Direct ragsmall search (vector_store_small) - Skip classification")

            try:
                # 🕐 Search trong ragsmall với k=5
                ragsmall_search_start_time = time.time()
                print(f"🕐 [STREAM TIMER] RAGSMALL_SEARCH START: {time.strftime('%H:%M:%S', time.localtime(ragsmall_search_start_time))}")

                ragsmall_results = await self.vector_store_small.similarity_search_with_score(
                    query=query,
                    k=5
                )

                ragsmall_search_end_time = time.time()
                ragsmall_search_duration = ragsmall_search_end_time - ragsmall_search_start_time
                print(f"🕐 [STREAM TIMER] RAGSMALL_SEARCH END: {time.strftime('%H:%M:%S', time.localtime(ragsmall_search_end_time))} - Duration: {ragsmall_search_duration:.3f}s")

                if not ragsmall_results:
                    print(f"[STREAM RAGSMALL] No results found → Fallback to RAG_CHAT stream")

                    # 🕐 Fallback RAG_CHAT stream
                    fallback_rag_start_time = time.time()
                    print(f"🕐 [STREAM TIMER] RAG_CHAT_FALLBACK START: {time.strftime('%H:%M:%S', time.localtime(fallback_rag_start_time))}")

                    async for chunk in self.rag_chat_streaming.generate_response_stream(query, session_id):
                        if chunk.get("type") == "done":
                            chunk["route_used"] = "RAG_CHAT_FALLBACK"
                            chunk["ragsmall_reason"] = "No results found"

                            fallback_rag_end_time = time.time()
                            fallback_rag_duration = fallback_rag_end_time - fallback_rag_start_time
                            total_end_time = time.time()
                            total_duration = total_end_time - total_start_time

                            chunk["timing_info"] = {
                                "total_duration": total_duration,
                                "ragsmall_search_duration": ragsmall_search_duration,
                                "rag_chat_duration": fallback_rag_duration
                            }

                            print(f"🕐 [STREAM TIMER] RAG_CHAT_FALLBACK END: Duration: {fallback_rag_duration:.3f}s")
                            print(f"🕐 [STREAM TIMER] TOTAL END: Total Duration: {total_duration:.3f}s")

                        yield chunk
                    return

                # 🕐 Bước 2: Check threshold với best result
                threshold_check_start_time = time.time()
                print(f"🕐 [STREAM TIMER] THRESHOLD_CHECK START: {time.strftime('%H:%M:%S', time.localtime(threshold_check_start_time))}")

                # Sort by similarity và lấy best
                ragsmall_results.sort(key=lambda x: x[1], reverse=True)
                best_doc, best_similarity = ragsmall_results[0]

                print(f"[STREAM RAGSMALL] Best similarity: {best_similarity:.3f}")
                print(f"[STREAM RAGSMALL] Threshold check: {best_similarity:.3f} >= 0.8?")

                if best_similarity >= 0.8:
                    # Stream answer từ ragsmall
                    print(f"[STREAM SUCCESS] ragsmall match found (similarity: {best_similarity:.3f})")

                    answer = best_doc.metadata.get('answer', '')
                    matched_question = best_doc.page_content
                    source = best_doc.metadata.get('source', '')
                    category = best_doc.metadata.get('category', '')

                    threshold_check_end_time = time.time()
                    threshold_check_duration = threshold_check_end_time - threshold_check_start_time
                    print(f"🕐 [STREAM TIMER] THRESHOLD_CHECK END: Duration: {threshold_check_duration:.3f}s")

                    # ✅ TRUE STREAMING answer theo chunks (faster than word-by-word)
                    chunk_size = 10  # 10 characters per chunk for smoother streaming
                    for i in range(0, len(answer), chunk_size):
                        chunk_text = answer[i:i + chunk_size]
                        yield {
                            "type": "chunk",
                            "content": chunk_text,
                            "route_used": "RAGSMALL_MATCH",
                            "timestamp": time.time()
                        }
                        await asyncio.sleep(0.02)  # Faster streaming - 20ms delay

                    # 🕐 TOTAL END - SUCCESS
                    total_end_time = time.time()
                    total_duration = total_end_time - total_start_time
                    print(f"🕐 [STREAM TIMER] TOTAL END: Total Duration: {total_duration:.3f}s")
                    print(f"🕐 [STREAM BREAKDOWN] Ragsmall Search: {ragsmall_search_duration:.3f}s | Threshold Check: {threshold_check_duration:.3f}s")

                    yield {
                        "type": "done",
                        "session_id": session_id or "ragsmall-session",
                        "route_used": "RAGSMALL_MATCH",
                        "ragsmall_info": {
                            "similarity_score": best_similarity,
                            "matched_question": matched_question,
                            "matched_category": category,
                            "source": source,
                            "total_results": len(ragsmall_results)
                        },
                        "timing_info": {
                            "total_duration": total_duration,
                            "ragsmall_search_duration": ragsmall_search_duration,
                            "threshold_check_duration": threshold_check_duration
                        },
                        "timestamp": time.time()
                    }
                    return

                threshold_check_end_time = time.time()
                threshold_check_duration = threshold_check_end_time - threshold_check_start_time
                print(f"🕐 [STREAM TIMER] THRESHOLD_CHECK END: Duration: {threshold_check_duration:.3f}s")

                # 🕐 Bước 3: Fallback to RAG_CHAT_STREAMING nếu similarity < 0.8
                print(f"[STREAM FALLBACK] ragsmall similarity < 0.8 ({best_similarity:.3f}) → RAG_CHAT_STREAMING")

                fallback_rag_start_time = time.time()
                print(f"🕐 [STREAM TIMER] RAG_CHAT_STREAMING_FALLBACK START: {time.strftime('%H:%M:%S', time.localtime(fallback_rag_start_time))}")

                async for chunk in self.rag_chat_streaming.generate_response_stream(query, session_id):
                    if chunk.get("type") == "done":
                        chunk["route_used"] = "RAG_CHAT_STREAMING_FALLBACK"
                        chunk["ragsmall_info"] = {
                            "best_similarity": best_similarity,
                            "total_results": len(ragsmall_results),
                            "best_category": best_doc.metadata.get('category', ''),
                            "threshold_reason": f"Similarity {best_similarity:.3f} < 0.8"
                        }

                        fallback_rag_end_time = time.time()
                        fallback_rag_duration = fallback_rag_end_time - fallback_rag_start_time
                        total_end_time = time.time()
                        total_duration = total_end_time - total_start_time

                        chunk["timing_info"] = {
                            "total_duration": total_duration,
                            "ragsmall_search_duration": ragsmall_search_duration,
                            "threshold_check_duration": threshold_check_duration,
                            "rag_chat_duration": fallback_rag_duration
                        }

                        print(f"🕐 [STREAM TIMER] RAG_CHAT_STREAMING_FALLBACK END: Duration: {fallback_rag_duration:.3f}s")
                        print(f"🕐 [STREAM TIMER] TOTAL END: Total Duration: {total_duration:.3f}s")
                        print(f"🕐 [STREAM BREAKDOWN] Ragsmall Search: {ragsmall_search_duration:.3f}s | Threshold Check: {threshold_check_duration:.3f}s | RAG_CHAT: {fallback_rag_duration:.3f}s")

                    yield chunk

            except Exception as e:
                print(f"[STREAM ERROR] ragsmall search failed: {e}")

                # 🕐 Fallback to RAG_CHAT stream on error
                error_fallback_start_time = time.time()
                print(f"🕐 [STREAM TIMER] RAG_CHAT_STREAMING_ERROR_FALLBACK START: {time.strftime('%H:%M:%S', time.localtime(error_fallback_start_time))}")

                async for chunk in self.rag_chat_streaming.generate_response_stream(query, session_id):
                    if chunk.get("type") == "done":
                        chunk["route_used"] = "RAG_CHAT_STREAMING_ERROR_FALLBACK"
                        chunk["ragsmall_error"] = str(e)

                        error_fallback_end_time = time.time()
                        error_fallback_duration = error_fallback_end_time - error_fallback_start_time
                        total_end_time = time.time()
                        total_duration = total_end_time - total_start_time

                        chunk["timing_info"] = {
                            "total_duration": total_duration,
                            "rag_chat_error_duration": error_fallback_duration
                        }

                        print(f"🕐 [STREAM TIMER] RAG_CHAT_STREAMING_ERROR_FALLBACK END: Duration: {error_fallback_duration:.3f}s")
                        print(f"🕐 [STREAM TIMER] TOTAL END: Total Duration: {total_duration:.3f}s")

                    yield chunk

        except Exception as e:
            logger.error(f"Error in master chatbot stream: {e}")
            yield {
                "type": "error",
                "content": "🤖 Xin lỗi, có lỗi xảy ra. Bạn vui lòng thử lại sau.",
                "timestamp": time.time()
            }
