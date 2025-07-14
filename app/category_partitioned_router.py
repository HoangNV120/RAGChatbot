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
    
    def __init__(self, use_categorized_data=True):
        # Initialize LLM cho classification
        self.llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        
        # Initialize embeddings cho vector search
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model="text-embedding-ada-002"
        )
        
        # Load data - ưu tiên data có category
        self.use_categorized_data = use_categorized_data
        self.data = self._load_data()
        
        # Similarity threshold cho vector search
        self.similarity_threshold = 0.9
        
        # Định nghĩa các category hợp lệ
        self.valid_categories = [
            "HỌC PHÍ", "NGÀNH HỌC", "QUY CHẾ THI", "ĐIỂM SỐ", 
            "DỊCH VỤ SINH VIÊN", "CƠ SỞ VẬT CHẤT", "CHƯƠNG TRÌNH HỌC"
        ]
        
        # Cache cho embeddings theo category
        self.category_embeddings = {}  # {category: numpy_array}
        self.category_questions_data = {}  # {category: [question_data]}
        
        # Cache cho classification results để tăng tốc độ và consistency
        self.classification_cache = {}  # {query_hash: classification_result}
        
        print(f"CategoryPartitionedRouter initialized with {len(self.data)} questions")
        print(f"Valid categories: {self.valid_categories}")
        
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

🎯 QUY TRÌNH PHÂN LOẠI (Thực hiện tuần tự):

BƯỚC 1: Xác định từ khóa chính trong câu hỏi
BƯỚC 2: Đối chiếu với 7 danh mục trên (ưu tiên theo thứ tự)
BƯỚC 3: Nếu không khớp → chọn "KHÁC"
BƯỚC 4: Double-check kết quả

⚠️ QUY TẮC VÀNG:
- CHỈ trả về TÊN DANH MỤC (viết hoa, không dấu phẩy, không giải thích)
- Nếu có thể thuộc 2 danh mục → chọn danh mục chính yếu nhất
- Khi nghi ngờ → chọn "KHÁC"
- KHÔNG bao giờ tự tạo danh mục mới

📝 MẪU PHẢN HỒI ĐÚNG:
- Input: "Học phí ngành CNTT?" → Output: "HỌC PHÍ"
- Input: "Thời tiết hôm nay?" → Output: "KHÁC"
- Input: "Lịch thi cuối kỳ?" → Output: "QUY CHẾ THI"

CHỈ TRẢ VỀ TÊN DANH MỤC DUY NHẤT - KHÔNG GIẢI THÍCH GÌ THÊM."""

        try:
            # 🎯 FEW-SHOT EXAMPLES để tăng consistency
            few_shot_examples = [
                ("Học phí ngành Công nghệ thông tin bao nhiêu?", "HỌC PHÍ"),
                ("Các ngành học tại trường có gì?", "NGÀNH HỌC"),
                ("Điều kiện thi tốt nghiệp là gì?", "QUY CHẾ THI"),
                ("Thang điểm tại trường như thế nào?", "ĐIỂM SỐ"),
                ("Làm sao để đăng ký học phần?", "DỊCH VỤ SINH VIÊN"),
                ("Thư viện trường có mở cửa không?", "CƠ SỞ VẬT CHẤT"),
                ("Lịch học môn Toán là gì?", "CHƯƠNG TRÌNH HỌC"),
                ("Hôm nay trời đẹp quá!", "KHÁC")
            ]
            
            # Tạo few-shot prompt
            few_shot_prompt = "\n".join([
                f"Ví dụ: \"{ex[0]}\" → {ex[1]}" for ex in few_shot_examples
            ])
            
            # 🧠 ENHANCED PROMPT với few-shot và reasoning
            enhanced_system_prompt = system_prompt + f"""

🎓 CÁC VÍ DỤ CHUẨN (học từ các trường hợp này):
{few_shot_prompt}

🔍 QUY TRÌNH TƒRANG LOGIC:
1. Đọc câu hỏi → Tìm từ khóa chính
2. So sánh với 8 ví dụ trên → Tìm mẫu tương tự
3. Áp dụng logic tương tự → Đưa ra quyết định
4. Kiểm tra lại → Đảm bảo đúng format

⚡ LƯU Ý QUAN TRỌNG:
- Phân tích CHÍNH XÁC như các ví dụ trên
- Phản hồi ĐỒNG NHẤT với training pattern
- KHÔNG thay đổi cách phân loại so với examples"""

            messages = [
                SystemMessage(content=enhanced_system_prompt),
                HumanMessage(content=f"Phân loại câu hỏi: \"{query}\"")
            ]
            
            # 🚀 DOUBLE-CHECK CLASSIFICATION với temperature=0 để consistency
            response = await self.llm.agenerate([messages])
            category = response.generations[0][0].text.strip().upper()
            
            # 🎯 VALIDATION & NORMALIZATION
            # Loại bỏ các ký tự không mong muốn
            category = category.replace(".", "").replace(",", "").replace(":", "").strip()
            
            # 📝 KEYWORD-BASED FALLBACK để đảm bảo consistency
            query_lower = query.lower()
            keyword_mapping = {
                "HỌC PHÍ": ["học phí", "chi phí", "tiền", "đóng học", "miễn giảm", "học bổng", "phí"],
                "NGÀNH HỌC": ["ngành", "chuyên ngành", "khoa", "đào tạo", "liên thông", "chuyên môn"],
                "QUY CHẾ THI": ["thi", "kiểm tra", "tốt nghiệp", "quy chế", "điều kiện", "quy định"],
                "ĐIỂM SỐ": ["điểm", "gpa", "thang điểm", "xếp loại", "học lực", "đánh giá"],
                "DỊCH VỤ SINH VIÊN": ["đăng ký", "thủ tục", "hỗ trợ", "dịch vụ", "tư vấn", "hành chính"],
                "CƠ SỞ VẬT CHẤT": ["phòng", "thư viện", "ký túc xá", "cơ sở", "trang thiết bị", "khuôn viên"],
                "CHƯƠNG TRÌNH HỌC": ["môn học", "tín chỉ", "lịch học", "thời khóa biểu", "chương trình", "khung"]
            }
            
            # 🔍 CONSISTENCY CHECK: So sánh LLM result với keyword matching
            keyword_category = None
            max_matches = 0
            
            for cat, keywords in keyword_mapping.items():
                matches = sum(1 for keyword in keywords if keyword in query_lower)
                if matches > max_matches:
                    max_matches = matches
                    keyword_category = cat
            
            # 🎯 FINAL DECISION với priority rules
            final_category = category
            
            # Rule 1: Nếu LLM classification không hợp lệ, dùng keyword fallback
            if category not in self.valid_categories and category != "KHÁC":
                if keyword_category and max_matches >= 1:
                    final_category = keyword_category
                    print(f"🔄 LLM classification '{category}' invalid, using keyword fallback: '{keyword_category}'")
                else:
                    final_category = "KHÁC"
                    print(f"🔄 LLM classification '{category}' invalid, no keywords found → KHÁC")
            
            # Rule 2: Cross-validation - nếu có conflict mạnh, ưu tiên keyword
            elif keyword_category and keyword_category != category and max_matches >= 2:
                print(f"🎯 Strong keyword evidence for '{keyword_category}' vs LLM '{category}', using keywords")
                final_category = keyword_category
            
            # 📊 LOG DECISION PROCESS
            print(f"🏷️  Classification result:")
            print(f"   LLM: {category}")
            print(f"   Keywords: {keyword_category} (matches: {max_matches})")
            print(f"   Final: {final_category}")
            
            # Kiểm tra category có hợp lệ không
            result = None
            if final_category in self.valid_categories:
                result = {
                    "category": final_category,
                    "is_valid": True,
                    "should_use_vector": True,
                    "llm_category": category,
                    "keyword_category": keyword_category,
                    "keyword_matches": max_matches
                }
            else:
                result = {
                    "category": "KHÁC",
                    "is_valid": False,
                    "should_use_vector": False,
                    "llm_category": category,
                    "keyword_category": keyword_category,
                    "keyword_matches": max_matches
                }
            
            # 💾 CACHE RESULT để consistency
            self.classification_cache[query_hash] = result
            
            return result
                
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            print(f"❌ LLM classification error: {e}")
            # Fallback: coi như category KHÁC
            return {
                "category": "KHÁC",
                "is_valid": False,
                "should_use_vector": False,
                "error": str(e)
            }
    
    async def _get_or_assign_category(self, question: str, row_index: int) -> str:
        """Lấy category từ data hoặc gán mới nếu chưa có (tối ưu hóa)"""
        
        # Nếu data đã có category column và có giá trị hợp lệ
        if 'category' in self.data.columns:
            existing_category = self.data.iloc[row_index].get('category', '')
            if pd.notna(existing_category) and str(existing_category).strip():
                category = str(existing_category).strip().upper()
                # Kiểm tra category có trong danh sách hợp lệ không
                if category in self.valid_categories:
                    return category
                elif category == "KHÁC":
                    return "KHÁC"
        
        # Nếu chưa có hoặc không hợp lệ, phân loại bằng LLM (chậm)
        print(f"🔄 Auto-categorizing question: {question[:50]}...")
        classification = await self._classify_query_category(question)
        return classification["category"]
    
    async def _initialize_category_embeddings(self, target_category: str):
        """Khởi tạo embeddings cho một category cụ thể"""
        
        if target_category in self.category_embeddings:
            return  # Đã khởi tạo rồi
        
        print(f"🔄 Initializing embeddings for category: {target_category}")
        
        # Lọc và chuẩn bị câu hỏi cho category này
        category_questions = []
        category_data = []
        
        for idx, row in self.data.iterrows():
            question = str(row['question']).strip()
            answer = str(row['answer']).strip()
            source = str(row.get('nguồn', '')).strip()
            
            if not question or not answer:
                continue
            
            # Lấy hoặc gán category
            row_category = await self._get_or_assign_category(question, idx)
            
            # Chỉ lấy câu hỏi thuộc category này
            if row_category == target_category:
                # Xử lý multiple questions (nếu có dấu |)
                question_parts = [q.strip() for q in question.split('|') if q.strip()]
                
                for q in question_parts:
                    category_questions.append(q)
                    category_data.append({
                        'original_index': idx,
                        'question': q,
                        'answer': answer,
                        'source': source,
                        'category': row_category
                    })
        
        if not category_questions:
            print(f"⚠️  No questions found for category: {target_category}")
            self.category_embeddings[target_category] = np.array([])
            self.category_questions_data[target_category] = []
            return
        
        print(f"📊 Processing {len(category_questions)} questions for category {target_category}")
        
        # Tạo embeddings cho category này
        try:
            # Batch processing để tối ưu
            batch_size = 50
            all_embeddings = []
            
            for i in range(0, len(category_questions), batch_size):
                batch = category_questions[i:i + batch_size]
                print(f"   Processing batch {i//batch_size + 1}/{(len(category_questions) + batch_size - 1)//batch_size}")
                
                # Get embeddings for batch
                batch_embeddings = await self.embeddings.aembed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.1)
            
            # Cache embeddings và data
            self.category_embeddings[target_category] = np.array(all_embeddings)
            self.category_questions_data[target_category] = category_data
            
            print(f"✅ Created embeddings for {len(category_questions)} questions in category {target_category}")
            print(f"📐 Embedding dimension: {self.category_embeddings[target_category].shape[1]}")
            
        except Exception as e:
            logger.error(f"Error creating embeddings for category {target_category}: {e}")
            print(f"❌ Error creating embeddings for category {target_category}: {e}")
            raise
    
    async def _get_query_embedding(self, query: str) -> np.ndarray:
        """Lấy embedding cho câu hỏi input"""
        try:
            embedding = await self.embeddings.aembed_query(query)
            return np.array(embedding)
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            raise
    
    def _find_most_similar_in_category(self, query_embedding: np.ndarray, category: str, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        """Tìm câu hỏi tương tự nhất trong một category cụ thể"""
        
        if category not in self.category_embeddings or len(self.category_embeddings[category]) == 0:
            return []
        
        category_embed = self.category_embeddings[category]
        category_data = self.category_questions_data[category]
        
        # Tính cosine similarity chỉ trong category này
        similarities = cosine_similarity([query_embedding], category_embed)[0]
        
        # Lấy top_k câu hỏi có similarity cao nhất
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity_score = similarities[idx]
            question_data = category_data[idx]
            
            results.append((idx, similarity_score, question_data))
        
        return results
    
    async def _vector_search_in_category(self, query: str, category: str) -> Dict:
        """Bước 3: Vector search trong category partition cụ thể"""
        
        # Khởi tạo embeddings cho category này
        await self._initialize_category_embeddings(category)
        
        print(f"🔍 Vector search in category '{category}' for query: {query[:50]}...")
        
        # Kiểm tra có data cho category này không
        if category not in self.category_questions_data or len(self.category_questions_data[category]) == 0:
            print(f"❌ No data found for category: {category}")
            return {
                "route": "RAG_CHAT",
                "reason": f"No questions found in category {category}",
                "similarity_score": 0.0
            }
        
        # Lấy embedding cho câu hỏi input
        query_embedding = await self._get_query_embedding(query)
        
        # Tìm câu hỏi tương tự nhất trong category này
        similar_questions = self._find_most_similar_in_category(query_embedding, category, top_k=3)
        
        if not similar_questions:
            print(f"❌ No similar questions found in category {category}")
            return {
                "route": "RAG_CHAT",
                "reason": f"No similar questions in category {category}",
                "similarity_score": 0.0
            }
        
        # Lấy câu hỏi có similarity cao nhất
        best_idx, best_similarity, best_question_data = similar_questions[0]
        
        print(f"📊 Best similarity in category '{category}': {best_similarity:.3f}")
        print(f"🔍 Best match: {best_question_data['question'][:50]}...")
        
        # Quyết định dựa trên threshold
        if best_similarity >= self.similarity_threshold:
            print(f"✅ Found good match in category '{category}' (similarity: {best_similarity:.3f} >= {self.similarity_threshold})")
            
            return {
                "route": "VECTOR_BASED",
                "similarity_score": best_similarity,
                "matched_question": best_question_data['question'],
                "answer": best_question_data['answer'],
                "source": best_question_data['source'],
                "matched_category": category,
                "all_matches": [
                    {
                        "question": q_data['question'],
                        "similarity": sim,
                        "answer": q_data['answer'][:100] + "..." if len(q_data['answer']) > 100 else q_data['answer'],
                        "category": category
                    }
                    for _, sim, q_data in similar_questions
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
                    "question": best_question_data['question'],
                    "similarity": best_similarity,
                    "answer": best_question_data['answer'][:100] + "..." if len(best_question_data['answer']) > 100 else best_question_data['answer'],
                    "category": category
                }
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
    
    def get_stats(self) -> Dict:
        """Thống kê về category-partitioned router"""
        
        total_questions = 0
        category_stats = {}
        
        for category in self.valid_categories:
            if category in self.category_questions_data:
                count = len(self.category_questions_data[category])
                category_stats[category] = count
                total_questions += count
            else:
                category_stats[category] = 0
        
        stats = {
            "total_questions": total_questions,
            "category_breakdown": category_stats,
            "similarity_threshold": self.similarity_threshold,
            "valid_categories": self.valid_categories,
            "partitions_initialized": list(self.category_embeddings.keys()),
            "use_categorized_data": self.use_categorized_data
        }
        
        return stats
    
    def set_similarity_threshold(self, threshold: float):
        """Điều chỉnh threshold cho vector search"""
        if 0.0 <= threshold <= 1.0:
            self.similarity_threshold = threshold
            print(f"📊 Vector similarity threshold updated to: {threshold}")
        else:
            print(f"❌ Invalid threshold: {threshold}. Must be between 0.0 and 1.0")
    
    async def test_query(self, query: str, show_details: bool = True) -> Dict:
        """Test một câu hỏi với category-partitioned approach"""
        
        print(f"\n🧪 Testing category-partitioned routing for: '{query}'")
        
        # Test classification
        print(f"\n📋 Step 1: LLM Classification")
        classification_result = await self._classify_query_category(query)
        category = classification_result['category']
        print(f"   Category: {category}")
        print(f"   Should use vector: {classification_result['should_use_vector']}")
        
        if classification_result['should_use_vector'] and category != "KHÁC":
            print(f"\n🔍 Step 2: Category-Partitioned Vector Search")
            
            # Initialize category embeddings
            await self._initialize_category_embeddings(category)
            
            if category in self.category_questions_data:
                category_count = len(self.category_questions_data[category])
                print(f"   📊 Searching in {category_count} questions in category '{category}'")
                
                # Get embedding và search
                query_embedding = await self._get_query_embedding(query)
                similar_questions = self._find_most_similar_in_category(query_embedding, category, top_k=5)
                
                if show_details and similar_questions:
                    print(f"\n📊 Top 5 similar questions in category '{category}':")
                    for i, (idx, similarity, q_data) in enumerate(similar_questions, 1):
                        print(f"  {i}. Similarity: {similarity:.3f}")
                        print(f"     Question: {q_data['question']}")
                        print(f"     Answer: {q_data['answer'][:100]}...")
                        print()
            else:
                print(f"   ❌ No data available for category '{category}'")
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

    async def test_classification_consistency(self, query: str, num_tests: int = 5) -> Dict:
        """Test tính nhất quán của classification cho cùng một câu hỏi"""
        
        print(f"\n🧪 Testing classification consistency for: '{query}'")
        print(f"🔄 Running {num_tests} classification attempts...")
        
        results = []
        categories = []
        
        for i in range(num_tests):
            print(f"   Test {i+1}/{num_tests}...", end="")
            
            try:
                result = await self._classify_query_category(query)
                category = result["category"]
                categories.append(category)
                results.append(result)
                print(f" → {category}")
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f" → ERROR: {e}")
                categories.append("ERROR")
        
        # Phân tích kết quả
        from collections import Counter
        category_counts = Counter(categories)
        total_valid = len([c for c in categories if c != "ERROR"])
        
        if total_valid == 0:
            return {
                "query": query,
                "consistency_rate": 0.0,
                "dominant_category": "ERROR",
                "all_results": categories,
                "is_consistent": False
            }
        
        # Tìm category xuất hiện nhiều nhất
        dominant_category = category_counts.most_common(1)[0][0]
        dominant_count = category_counts[dominant_category]
        consistency_rate = dominant_count / total_valid
        
        print(f"\n📊 Consistency Analysis:")
        print(f"   Dominant category: {dominant_category}")
        print(f"   Consistency rate: {consistency_rate:.1%} ({dominant_count}/{total_valid})")
        print(f"   All results: {dict(category_counts)}")
        
        is_consistent = consistency_rate >= 0.8  # 80% threshold
        consistency_status = "✅ CONSISTENT" if is_consistent else "❌ INCONSISTENT"
        print(f"   Status: {consistency_status}")
        
        return {
            "query": query,
            "consistency_rate": consistency_rate,
            "dominant_category": dominant_category,
            "category_distribution": dict(category_counts),
            "all_results": categories,
            "is_consistent": is_consistent,
            "num_tests": num_tests
        }

    async def batch_test_consistency(self, test_queries: List[str], tests_per_query: int = 3) -> Dict:
        """Test tính nhất quán cho nhiều câu hỏi"""
        
        print(f"\n🚀 Batch testing classification consistency")
        print(f"📝 {len(test_queries)} queries, {tests_per_query} tests each")
        
        all_results = []
        consistent_count = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📋 Query {i}/{len(test_queries)}: {query[:50]}...")
            
            result = await self.test_classification_consistency(query, tests_per_query)
            all_results.append(result)
            
            if result["is_consistent"]:
                consistent_count += 1
        
        # Tổng kết
        overall_consistency = consistent_count / len(test_queries)
        
        print(f"\n🎯 OVERALL CONSISTENCY REPORT:")
        print(f"   Total queries tested: {len(test_queries)}")
        print(f"   Consistent queries: {consistent_count}")
        print(f"   Overall consistency rate: {overall_consistency:.1%}")
        
        # Phân tích category performance
        category_performance = {}
        for result in all_results:
            cat = result["dominant_category"]
            if cat not in category_performance:
                category_performance[cat] = []
            category_performance[cat].append(result["consistency_rate"])
        
        print(f"\n📊 Category Performance:")
        for cat, rates in category_performance.items():
            avg_rate = sum(rates) / len(rates)
            print(f"   {cat}: {avg_rate:.1%} avg consistency ({len(rates)} queries)")
        
        return {
            "overall_consistency_rate": overall_consistency,
            "consistent_queries": consistent_count,
            "total_queries": len(test_queries),
            "category_performance": {
                cat: sum(rates) / len(rates) 
                for cat, rates in category_performance.items()
            },
            "detailed_results": all_results
        }
    
    def clear_classification_cache(self):
        """Xóa cache classification"""
        self.classification_cache.clear()
        print("🗑️ Classification cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Thống kê cache"""
        return {
            "cache_size": len(self.classification_cache),
            "cached_queries": list(self.classification_cache.keys())[:5]  # Show first 5 hashes
        }
    
    def save_classification_cache(self, filepath: str = "classification_cache.json"):
        """Lưu cache vào file để persistent consistency"""
        import json
        try:
            # Convert để serializable
            serializable_cache = {}
            for query_hash, result in self.classification_cache.items():
                serializable_cache[query_hash] = result
                
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_cache, f, ensure_ascii=False, indent=2)
            print(f"💾 Classification cache saved to {filepath} ({len(serializable_cache)} entries)")
        except Exception as e:
            print(f"❌ Error saving cache: {e}")
    
    def load_classification_cache(self, filepath: str = "classification_cache.json"):
        """Load cache từ file để maintain consistency"""
        import json
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.classification_cache = json.load(f)
            print(f"📥 Classification cache loaded from {filepath} ({len(self.classification_cache)} entries)")
            print("✅ This ensures 100% classification consistency for cached queries!")
        except FileNotFoundError:
            print(f"⚠️  Cache file {filepath} not found, starting with empty cache")
        except Exception as e:
            print(f"❌ Error loading cache: {e}")
