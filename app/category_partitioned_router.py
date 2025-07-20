from typing import Dict, Optional, List, Tuple
import logging
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from app.config import settings
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import asyncio

logger = logging.getLogger(__name__)

class CategoryPartitionedRouter:
    """
    Advanced Hybrid Router với category-based data partitioning
    
    Luồng hoạt động:
    1. LLM Classification để xác định category
    2. Nếu category = "KHÁC" → RAG_CHAT (bỏ qua vector)
    3. Nếu category hợp lệ → Vector search CHỈ trong partition tương ứng
    4. Tìm similarity trong partition, nếu đủ cao → trả answer, nếu không → RAG_CHAT
    """
    
    def __init__(self, vector_store=None, use_categorized_data=True):
        # Initialize LLM cho classification
        self.llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model_name="gpt-4o-mini",
            temperature=0
        )
        
        # Initialize embeddings cho vector search
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model="text-embedding-ada-002"
        )
        
        # Use existing vector store (Qdrant) instead of local cache
        self.vector_store = vector_store
        if not self.vector_store:
            raise ValueError("Vector store (Qdrant) is required for CategoryPartitionedRouter")
        
        # Load data for category information only (không cần embed lại)
        self.use_categorized_data = use_categorized_data
        self.data = self._load_data()
        
        # Similarity threshold cho vector search
        self.similarity_threshold = 0.8
        
        # Định nghĩa các category hợp lệ
        self.valid_categories = [
            "HỌC PHÍ", "NGÀNH HỌC", "QUY CHẾ THI", "ĐIỂM SỐ", 
            "DỊCH VỤ SINH VIÊN", "CƠ SỞ VẬT CHẤT", "CHƯƠNG TRÌNH HỌC"
        ]
        
        # Cache cho classification results để tăng tốc độ và consistency
        self.classification_cache = {}  # {query_hash: classification_result}
        
        # Cache cho vector search results (optional)
        self.vector_search_cache = {}  # {query_hash: vector_search_result}
        self.cache_expiry_time = 3600  # 1 hour cache expiry
        
        print(f"CategoryPartitionedRouter initialized with {len(self.data)} questions")
        print(f"Valid categories: {self.valid_categories}")
        print(f"✅ Using existing Qdrant vector store for category-partitioned search")
        
        # Hiển thị thống kê category
        if 'category' in self.data.columns:
            category_stats = self.data['category'].value_counts()
            print(f"📊 Category distribution:")
            for cat, count in category_stats.items():
                print(f"   {cat}: {count} questions")
        else:
            print("⚠️  No category column found - will categorize on demand")
    
    def _load_data(self):
        """Load data với ưu tiên file có category"""
        try:
            if self.use_categorized_data:
                # Thử load file đã có category trước
                try:
                    df = pd.read_excel('app/data_test_with_categories.xlsx')
                    print("✅ Loaded categorized data file")
                    return df
                except FileNotFoundError:
                    print("⚠️  Categorized data file not found, falling back to original")
            
            # Fallback sang file gốc
            df = pd.read_excel('app/data_test.xlsx')
            print("✅ Loaded original data file")
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise

    async def _classify_query_category(self, query: str) -> Dict:
        """Bước 1: Phân loại category bằng LLM với Prompt Engineering cao cấp"""
        
        # 🚀 CACHE CHECK để đảm bảo consistency
        import hashlib
        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
        
        if query_hash in self.classification_cache:
            cached_result = self.classification_cache[query_hash]
            print(f"🎯 Using cached classification: {cached_result['category']}")
            return cached_result
        
        system_prompt = f"""Bạn là chuyên gia phân loại câu hỏi về trường đại học với độ chính xác 100%. Nhiệm vụ của bạn là phân loại câu hỏi vào ĐÚNG MỘT trong các danh mục sau:

📚 DANH MỤC CỤ THỂ (7 loại):
1. HỌC PHÍ - Tất cả về tiền bạc:
   • Học phí, chi phí học tập, tiền đóng học
   • Miễn giảm học phí, học bổng
   • Từ khóa: "học phí", "chi phí", "tiền", "đóng học", "miễn giảm", "học bổng"

2. NGÀNH HỌC - Về chuyên ngành:
   • Các ngành học, chuyên ngành, khoa
   • Chương trình đào tạo, liên thông
   • Từ khóa: "ngành", "chuyên ngành", "khoa", "đào tạo", "liên thông"

3. QUY CHẾ THI - Về thi cử và tốt nghiệp:
   • Quy định thi, điều kiện thi, lịch thi
   • Điều kiện tốt nghiệp, quy chế học vụ
   • Từ khóa: "thi", "kiểm tra", "tốt nghiệp", "quy chế", "điều kiện"

4. ĐIỂM SỐ - Về điểm và đánh giá:
   • Thang điểm, cách tính điểm, GPA
   • Xếp loại, học lực, điểm trung bình
   • Từ khóa: "điểm", "GPA", "thang điểm", "xếp loại", "học lực"

5. DỊCH VỤ SINH VIÊN - Về thủ tục và hỗ trợ:
   • Thủ tục hành chính, đăng ký học phần
   • Dịch vụ hỗ trợ, tư vấn sinh viên
   • Từ khóa: "đăng ký", "thủ tục", "hỗ trợ", "dịch vụ", "tư vấn"

6. CƠ SỞ VẬT CHẤT - Về không gian vật lý:
   • Phòng học, thư viện, ký túc xá
   • Cơ sở vật chất, trang thiết bị
   • Từ khóa: "phòng", "thư viện", "ký túc xá", "cơ sở", "trang thiết bị"

7. CHƯƠNG TRÌNH HỌC - Về môn học và lịch học:
   • Môn học, tín chỉ, thời khóa biểu
   • Lịch học, lịch thi, khung chương trình
   • Từ khóa: "môn học", "tín chỉ", "lịch học", "thời khóa biểu"

🚫 KHÁC - Không liên quan giáo dục:
   • Thời tiết, nấu ăn, thể thao, giải trí
   • Tin tức, công nghệ không liên quan học tập
   • Câu hỏi cá nhân không về trường học

CHỈ TRẢ VỀ TÊN DANH MỤC DUY NHẤT - KHÔNG GIẢI THÍCH GÌ THÊM."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Phân loại câu hỏi: \"{query}\"")
            ]
            
            response = await self.llm.agenerate([messages])
            category = response.generations[0][0].text.strip().upper()
            category = category.replace(".", "").replace(",", "").replace(":", "").strip()
            
            # Validate category
            if category in self.valid_categories:
                result = {
                    "category": category,
                    "is_valid": True,
                    "should_use_vector": True
                }
            else:
                result = {
                    "category": "KHÁC",
                    "is_valid": False,
                    "should_use_vector": False
                }
            
            # Cache result
            self.classification_cache[query_hash] = result
            return result
                
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            return {
                "category": "KHÁC",
                "is_valid": False,
                "should_use_vector": False,
                "error": str(e)
            }

    async def _vector_search_in_category(self, query: str, category: str) -> Dict:
        """Bước 3: Vector search trong category partition với post-filtering"""
        
        print(f"🔍 Vector search in category '{category}' using Qdrant for query: {query[:50]}...")
        
        try:
            # Thực hiện similarity search sử dụng VectorStore wrapper với async method
            results = await self.vector_store.similarity_search_with_score(
                query=query,
                k=50  # Lấy nhiều results để filter
            )
            
            if not results:
                print(f"❌ No results found from Qdrant")
                return {
                    "route": "RAG_CHAT",
                    "reason": f"No similar documents found in vector store",
                    "similarity_score": 0.0,
                    "searched_category": category
                }
            
            # Filter results theo category với improved matching
            category_results = []
            found_categories = {}
            
            for doc, similarity in results:
                doc_category = doc.metadata.get('category', '').strip().upper()
                
                # Track tất cả categories để debug
                if doc_category in found_categories:
                    found_categories[doc_category] += 1
                else:
                    found_categories[doc_category] = 1
                
                # Improved category matching
                if doc_category == category or \
                   doc_category.replace(' ', '') == category.replace(' ', '') or \
                   (doc_category and category and doc_category in category) or \
                   (doc_category and category and category in doc_category):
                    category_results.append((doc, similarity))
            
            print(f"📊 Categories found in Qdrant results: {found_categories}")
            
            if not category_results:
                print(f"❌ No results found in category '{category}' after filtering")
                print(f"📊 Total results from Qdrant: {len(results)}")
                print(f"🔍 Available categories: {list(found_categories.keys())}")
                
                # Try fuzzy matching with available categories
                best_match_category = None
                for available_cat in found_categories.keys():
                    if available_cat and category:
                        # Simple fuzzy matching
                        if category.replace(' ', '').lower() in available_cat.replace(' ', '').lower() or \
                           available_cat.replace(' ', '').lower() in category.replace(' ', '').lower():
                            best_match_category = available_cat
                            break
                
                if best_match_category:
                    print(f"� Trying fuzzy match with category '{best_match_category}'")
                    for doc, similarity in results:
                        doc_category = doc.metadata.get('category', '').strip().upper()
                        if doc_category == best_match_category:
                            category_results.append((doc, similarity))
                
                if not category_results:
                    return {
                        "route": "RAG_CHAT",
                        "reason": f"No documents found in category '{category}' (found {len(results)} total results in categories: {list(found_categories.keys())})",
                        "similarity_score": 0.0,
                        "searched_category": category,
                        "available_categories": list(found_categories.keys())
                    }
            
            # Sort filtered results by similarity
            category_results.sort(key=lambda x: x[1], reverse=True)
            
            # Lấy kết quả tốt nhất trong category
            best_doc, best_similarity = category_results[0]
            
            print(f"📊 Found {len(category_results)} results in category '{category}'")
            print(f"📊 Best similarity in category '{category}': {best_similarity:.3f}")
            print(f"🔍 Best match: {best_doc.page_content[:50]}...")
            
            # Quyết định dựa trên threshold
            if best_similarity >= self.similarity_threshold:
                print(f"✅ Found good match in category '{category}' (similarity: {best_similarity:.3f} >= {self.similarity_threshold})")
                
                # Extract metadata
                metadata = best_doc.metadata
                answer = metadata.get('answer', '')
                source = metadata.get('source', '')
                matched_question = best_doc.page_content
                
                return {
                    "route": "VECTOR_BASED",
                    "similarity_score": best_similarity,
                    "matched_question": matched_question,
                    "answer": answer,
                    "source": source,
                    "matched_category": category,
                    "all_matches": [
                        {
                            "question": doc.page_content,
                            "similarity": sim,
                            "answer": doc.metadata.get('answer', '')[:100] + "..." if len(doc.metadata.get('answer', '')) > 100 else doc.metadata.get('answer', ''),
                            "category": category
                        }
                        for doc, sim in category_results[:5]  # Top 5 in category
                    ]
                }
            else:
                print(f"❌ Similarity too low in category '{category}' (best: {best_similarity:.3f} < {self.similarity_threshold})")
                
                return {
                    "route": "RAG_CHAT",
                    "reason": f"Best similarity {best_similarity:.3f} in category '{category}' below threshold {self.similarity_threshold}",
                    "similarity_score": best_similarity,
                    "searched_category": category,
                    "best_match": {
                        "question": best_doc.page_content,
                        "similarity": best_similarity,
                        "answer": best_doc.metadata.get('answer', '')[:100] + "..." if len(best_doc.metadata.get('answer', '')) > 100 else best_doc.metadata.get('answer', ''),
                        "category": category
                    }
                }
                
        except Exception as e:
            logger.error(f"Error in Qdrant search for category {category}: {e}")
            print(f"❌ Error in Qdrant search for category '{category}': {e}")
            return {
                "route": "RAG_CHAT",
                "reason": f"Error during Qdrant search in category {category}: {str(e)}",
                "similarity_score": 0.0,
                "searched_category": category
            }

    async def route_query(self, query: str) -> Dict:
        """Main routing function với category-partitioned approach"""
        try:
            print(f"\n🚀 Category-partitioned routing for query: {query[:50]}...")
            
            # Bước 1: LLM Classification
            classification_result = await self._classify_query_category(query)
            category = classification_result["category"]
            should_use_vector = classification_result["should_use_vector"]
            
            # Bước 2: Quyết định luồng dựa trên category
            if not should_use_vector or category == "KHÁC":
                print(f"⚡ Category '{category}' → Skip vector search → Direct RAG_CHAT")
                return {
                    "route": "RAG_CHAT",
                    "reason": f"Category '{category}' requires full RAG processing",
                    "query": query,
                    "classification": classification_result,
                    "similarity_score": 0.0
                }
            
            # Bước 3: Vector Search trong category partition cụ thể
            print(f"🎯 Category '{category}' → Search in category partition...")
            vector_result = await self._vector_search_in_category(query, category)
            
            # Thêm thông tin classification vào kết quả
            vector_result["query"] = query
            vector_result["classification"] = classification_result
            
            return vector_result
            
        except Exception as e:
            logger.error(f"Error in category-partitioned routing: {e}")
            print(f"❌ Error in category-partitioned routing: {e}")
            return {
                "route": "RAG_CHAT",
                "reason": f"Error during routing: {str(e)}",
                "query": query,
                "similarity_score": 0.0
            }

    async def test_query(self, query: str, show_details: bool = True) -> Dict:
        """Test một câu hỏi với category-partitioned approach sử dụng Qdrant"""
        
        print(f"\n🧪 Testing category-partitioned routing (Qdrant) for: '{query}'")
        
        # Test classification
        print(f"\n📋 Step 1: LLM Classification")
        classification_result = await self._classify_query_category(query)
        category = classification_result['category']
        print(f"   Category: {category}")
        print(f"   Should use vector: {classification_result['should_use_vector']}")
        
        if classification_result['should_use_vector'] and category != "KHÁC":
            print(f"\n🔍 Step 2: Category-Filtered Qdrant Search")
            
            try:
                # Test query Qdrant sử dụng VectorStore wrapper
                results = await self.vector_store.similarity_search_with_score(
                    query=query,
                    k=1
                )
                
                # Filter by category
                category_results = []
                for doc, sim in results:
                    if doc.metadata.get('category', '').strip().upper() == category:
                        category_results.append((doc, sim))
                
                print(f"   📊 Found {len(category_results)} results in category '{category}' from Qdrant (total: {len(results)})")
                
                if show_details and category_results:
                    print(f"\n📊 Top {min(5, len(category_results))} similar questions in category '{category}':")
                    for i, (doc, similarity) in enumerate(category_results[:5], 1):
                        print(f"  {i}. Similarity: {similarity:.3f}")
                        print(f"     Question: {doc.page_content}")
                        answer = doc.metadata.get('answer', '')
                        print(f"     Answer: {answer[:100]}...")
                        print()
            except Exception as e:
                print(f"   ❌ Error querying Qdrant: {e}")
        else:
            print(f"\n⚡ Step 2: Skipped vector search (Category: {category})")
        
        # Get final routing result
        print(f"\n🎯 Final Routing Result:")
        result = await self.route_query(query)
        
        print(f"   Route: {result['route']}")
        if result['route'] == "VECTOR_BASED":
            print(f"   ✅ Matched answer: {result['answer'][:150]}...")
            print(f"   📊 Similarity: {result['similarity_score']:.3f}")
            print(f"   🏷️  Matched in category: {result.get('matched_category', 'N/A')}")
        else:
            print(f"   ❌ Reason: {result['reason']}")
        
        return result

    def get_stats(self):
        """Trả về thống kê về router performance và cache"""
        stats = {
            "vector_store_type": "Qdrant (CategoryPartitioned)",
            "total_questions": len(self.data) if hasattr(self, 'data') and self.data is not None else 0,
            "similarity_threshold": self.similarity_threshold,
            "valid_categories": self.valid_categories.copy(),
            "classification_cache_size": len(self.classification_cache),
            "vector_search_cache_size": len(self.vector_search_cache),
            "categorized_data_loaded": self.use_categorized_data
        }
        
        # Thống kê category từ data nếu có
        if hasattr(self, 'data') and self.data is not None and 'category' in self.data.columns:
            category_stats = self.data['category'].value_counts().to_dict()
            stats["category_breakdown"] = category_stats
        else:
            stats["category_breakdown"] = {}
        
        # Thống kê cache theo category
        cache_by_category = {}
        for cached_result in self.classification_cache.values():
            category = cached_result.get('category', 'Unknown')
            cache_by_category[category] = cache_by_category.get(category, 0) + 1
        
        stats["classification_cache_by_category"] = cache_by_category
        
        return stats

    async def debug_vector_store_categories(self, sample_size=10):
        """Debug method để kiểm tra category trong Qdrant"""
        print(f"\n🔍 Debugging Qdrant vector store categories...")
        
        try:
            # Lấy sample documents từ Qdrant sử dụng VectorStore wrapper
            results = await self.vector_store.similarity_search_with_score(
                query="test query",
                k=sample_size
            )
            
            print(f"📊 Found {len(results)} documents in Qdrant:")
            
            categories_found = {}
            for i, (doc, score) in enumerate(results):
                metadata = doc.metadata
                category = metadata.get('category', 'NO_CATEGORY')
                source = metadata.get('source', 'NO_SOURCE') 
                
                print(f"  {i+1}. Category: '{category}' | Source: '{source}'")
                print(f"     Content: {doc.page_content[:50]}...")
                print(f"     Metadata keys: {list(metadata.keys())}")
                
                if category in categories_found:
                    categories_found[category] += 1
                else:
                    categories_found[category] = 1
            
            print(f"\n📋 Category Summary in Qdrant:")
            for category, count in categories_found.items():
                print(f"   '{category}': {count} documents")
                
            return categories_found
            
        except Exception as e:
            print(f"❌ Error debugging vector store: {e}")
            return {}

    async def analyze_text_format_issue(self):
        """Phân tích vấn đề format text giữa cache và Qdrant"""
        print(f"\n🔍 Analyzing Text Format Issues...")
        
        try:
            # Lấy sample documents từ Qdrant sử dụng VectorStore wrapper
            sample_results = await self.vector_store.similarity_search_with_score(
                query="test query",
                k=5
            )
            
            print(f"📊 Sample document formats in Qdrant:")
            for i, (doc, score) in enumerate(sample_results):
                print(f"\n  Document {i+1}:")
                print(f"    Score: {score:.4f}")
                print(f"    Category: '{doc.metadata.get('category', 'NO_CATEGORY')}'")
                print(f"    Content length: {len(doc.page_content)} characters")
                print(f"    Content format:")
                
                # Show first 200 chars with format analysis
                content_preview = doc.page_content[:200]
                print(f"    '{content_preview}...'")
                
                # Analyze format
                if content_preview.startswith("Question:"):
                    print(f"    ✅ Format: Question-Answer format")
                    
                    # Extract just the question part
                    try:
                        lines = doc.page_content.split('\n')
                        question_line = lines[0].replace("Question: ", "").strip()
                        answer_line = lines[1].replace("Answer: ", "").strip() if len(lines) > 1 else ""
                        
                        print(f"    📝 Extracted Question: '{question_line[:100]}...'")
                        print(f"    💬 Extracted Answer: '{answer_line[:100]}...'")
                        
                    except Exception as e:
                        print(f"    ❌ Error parsing Q&A format: {e}")
                else:
                    print(f"    ⚠️  Format: Unknown format")
            
            return True
            
        except Exception as e:
            print(f"❌ Error analyzing text format: {e}")
            return False
