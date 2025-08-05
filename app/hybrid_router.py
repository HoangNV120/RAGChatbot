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

class HybridRouter:
    """
    Hybrid Router kết hợp LLM Classification + Vector Similarity
    
    Luồng hoạt động:
    1. LLM Classification để xác định category
    2. Nếu category = "KHÁC" → RAG_CHAT (bỏ qua vector)
    3. Nếu category thuộc các category định trước → Vector Router
    4. Vector Router: tìm similarity, nếu đủ cao → trả answer, nếu không → RAG_CHAT
    """
    
    def __init__(self):
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
        
        # Load data từ data_test.xlsx
        self.data = pd.read_excel('app/data_test.xlsx')
        
        # Similarity threshold cho vector search
        self.similarity_threshold = 0.75
        
        # Cache cho embeddings
        self.question_embeddings = None
        self.questions_data = []
        
        # Định nghĩa các category hợp lệ
        self.valid_categories = [
            "HỌC PHÍ", "NGÀNH HỌC", "QUY CHẾ THI", "ĐIỂM SỐ", 
            "DỊCH VỤ SINH VIÊN", "CƠ SỞ VẬT CHẤT", "CHƯƠNG TRÌNH HỌC"
        ]
        
        print(f"HybridRouter initialized with {len(self.data)} questions")
        print(f"Valid categories: {self.valid_categories}")
        
    async def _classify_query_category(self, query: str) -> Dict:
        """Bước 1: Phân loại category bằng LLM"""
        
        system_prompt = f"""Bạn là một hệ thống phân loại câu hỏi về trường đại học. Hãy phân loại câu hỏi vào một trong các danh mục sau:

DANH MỤC HỢP LỆ:
- HỌC PHÍ: Câu hỏi về chi phí học tập, học phí, miễn giảm học phí
- NGÀNH HỌC: Câu hỏi về các ngành học, chuyên ngành, chương trình đào tạo
- QUY CHẾ THI: Câu hỏi về quy định thi cử, điều kiện thi, lịch thi
- ĐIỂM SỐ: Câu hỏi về điểm số, thang điểm, cách tính điểm
- DỊCH VỤ SINH VIÊN: Câu hỏi về dịch vụ hỗ trợ sinh viên, thủ tục hành chính
- CƠ SỞ VẬT CHẤT: Câu hỏi về cơ sở vật chất, phòng học, thư viện, ký túc xá
- CHƯƠNG TRÌNH HỌC: Câu hỏi về chương trình học, môn học, thời khóa biểu

QUAN TRỌNG:
- Nếu câu hỏi KHÔNG thuộc các danh mục trên hoặc không liên quan đến trường đại học, hãy trả về "KHÁC"
- Chỉ trả về TÊN DANH MỤC, không giải thích thêm
- Ví dụ câu hỏi thuộc "KHÁC": thời tiết, nấu ăn, thể thao, tin tức, công nghệ không liên quan giáo dục...

CHỈ TRẢ VỀ TÊN DANH MỤC DUY NHẤT."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Câu hỏi: {query}")
            ]
            
            response = await self.llm.agenerate([messages])
            category = response.generations[0][0].text.strip().upper()
            
            # Kiểm tra category có hợp lệ không
            if category in self.valid_categories:
                print(f"🏷️  Category classified: {category}")
                return {
                    "category": category,
                    "is_valid": True,
                    "should_use_vector": True
                }
            else:
                print(f"🏷️  Category classified: {category} (INVALID - treating as KHÁC)")
                return {
                    "category": "KHÁC",
                    "is_valid": False,
                    "should_use_vector": False
                }
                
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
    
    async def _initialize_embeddings(self):
        """Khởi tạo embeddings cho vector search (giống VectorRouter)"""
        
        if self.question_embeddings is not None:
            return  # Đã khởi tạo rồi
            
        print("🔄 Initializing question embeddings for vector search...")
        
        # Chuẩn bị danh sách câu hỏi
        questions = []
        self.questions_data = []
        
        for idx, row in self.data.iterrows():
            question = str(row['question']).strip()
            answer = str(row['answer']).strip()
            source = str(row.get('nguồn', '')).strip()
            
            if question and answer:  # Chỉ lấy câu hỏi có answer
                # Xử lý multiple questions (nếu có dấu |)
                question_parts = [q.strip() for q in question.split('|') if q.strip()]
                
                for q in question_parts:
                    questions.append(q)
                    self.questions_data.append({
                        'original_index': idx,
                        'question': q,
                        'answer': answer,
                        'source': source
                    })
        
        print(f"📊 Processing {len(questions)} questions for embedding...")
        
        # Tạo embeddings cho tất cả câu hỏi
        try:
            # Batch processing để tối ưu
            batch_size = 50
            all_embeddings = []
            
            for i in range(0, len(questions), batch_size):
                batch = questions[i:i + batch_size]
                print(f"   Processing batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")
                
                # Get embeddings for batch
                batch_embeddings = await self.embeddings.aembed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.1)
            
            # Convert to numpy array
            self.question_embeddings = np.array(all_embeddings)
            
            print(f"✅ Successfully created embeddings for {len(questions)} questions")
            print(f"📐 Embedding dimension: {self.question_embeddings.shape[1]}")
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            print(f"❌ Error creating embeddings: {e}")
            raise
    
    async def _get_query_embedding(self, query: str) -> np.ndarray:
        """Lấy embedding cho câu hỏi input"""
        try:
            embedding = await self.embeddings.aembed_query(query)
            return np.array(embedding)
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            raise
    
    def _find_most_similar_question(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        """Tìm câu hỏi tương tự nhất trong database"""
        
        if self.question_embeddings is None:
            return []
        
        # Tính cosine similarity
        similarities = cosine_similarity([query_embedding], self.question_embeddings)[0]
        
        # Lấy top_k câu hỏi có similarity cao nhất
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity_score = similarities[idx]
            question_data = self.questions_data[idx]
            
            results.append((idx, similarity_score, question_data))
        
        return results
    
    async def _vector_search(self, query: str) -> Dict:
        """Bước 3: Vector search (giống VectorRouter)"""
        
        # Khởi tạo embeddings nếu chưa có
        await self._initialize_embeddings()
        
        print(f"🔍 Vector search for query: {query[:50]}...")
        
        # Lấy embedding cho câu hỏi input
        query_embedding = await self._get_query_embedding(query)
        
        # Tìm câu hỏi tương tự nhất
        similar_questions = self._find_most_similar_question(query_embedding, top_k=3)
        
        if not similar_questions:
            print("❌ No similar questions found in vector search")
            return {
                "route": "RAG_CHAT",
                "reason": "No questions in database",
                "similarity_score": 0.0
            }
        
        # Lấy câu hỏi có similarity cao nhất
        best_idx, best_similarity, best_question_data = similar_questions[0]
        
        print(f"📊 Best vector similarity: {best_similarity:.3f}")
        print(f"🔍 Best match: {best_question_data['question'][:50]}...")
        
        # Quyết định dựa trên threshold
        if best_similarity >= self.similarity_threshold:
            print(f"✅ Found good vector match (similarity: {best_similarity:.3f} >= {self.similarity_threshold})")
            
            return {
                "route": "VECTOR_BASED",
                "similarity_score": best_similarity,
                "matched_question": best_question_data['question'],
                "answer": best_question_data['answer'],
                "source": best_question_data['source'],
                "all_matches": [
                    {
                        "question": q_data['question'],
                        "similarity": sim,
                        "answer": q_data['answer'][:100] + "..." if len(q_data['answer']) > 100 else q_data['answer']
                    }
                    for _, sim, q_data in similar_questions
                ]
            }
        else:
            print(f"❌ Vector similarity too low (best: {best_similarity:.3f} < {self.similarity_threshold})")
            
            return {
                "route": "RAG_CHAT",
                "reason": f"Vector similarity {best_similarity:.3f} below threshold {self.similarity_threshold}",
                "similarity_score": best_similarity,
                "best_match": {
                    "question": best_question_data['question'],
                    "similarity": best_similarity,
                    "answer": best_question_data['answer'][:100] + "..." if len(best_question_data['answer']) > 100 else best_question_data['answer']
                }
            }
    
    async def route_query(self, query: str) -> Dict:
        """Main routing function với hybrid approach"""
        try:
            print(f"\n🚀 Hybrid routing for query: {query[:50]}...")
            
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
            
            # Bước 3: Vector Search cho category hợp lệ
            print(f"🎯 Category '{category}' → Proceed to vector search...")
            vector_result = await self._vector_search(query)
            
            # Thêm thông tin classification vào kết quả
            vector_result["query"] = query
            vector_result["classification"] = classification_result
            
            return vector_result
            
        except Exception as e:
            logger.error(f"Error in hybrid routing: {e}")
            print(f"❌ Error in hybrid routing: {e}")
            return {
                "route": "RAG_CHAT",
                "reason": f"Error during hybrid routing: {str(e)}",
                "query": query,
                "similarity_score": 0.0
            }
    
    def get_stats(self) -> Dict:
        """Thống kê về hybrid router"""
        stats = {
            "total_questions": len(self.questions_data),
            "embedding_dimension": self.question_embeddings.shape[1] if self.question_embeddings is not None else 0,
            "similarity_threshold": self.similarity_threshold,
            "valid_categories": self.valid_categories,
            "vector_status": "ready" if self.question_embeddings is not None else "not_initialized"
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
        """Test một câu hỏi và hiển thị chi tiết toàn bộ luồng"""
        
        print(f"\n🧪 Testing hybrid routing for: '{query}'")
        
        # Test classification
        print(f"\n📋 Step 1: LLM Classification")
        classification_result = await self._classify_query_category(query)
        print(f"   Category: {classification_result['category']}")
        print(f"   Should use vector: {classification_result['should_use_vector']}")
        
        if classification_result['should_use_vector']:
            print(f"\n🔍 Step 2: Vector Search")
            await self._initialize_embeddings()
            
            # Get embedding
            query_embedding = await self._get_query_embedding(query)
            
            # Find similar questions
            similar_questions = self._find_most_similar_question(query_embedding, top_k=5)
            
            if show_details:
                print(f"\n📊 Top 5 similar questions:")
                for i, (idx, similarity, q_data) in enumerate(similar_questions, 1):
                    print(f"  {i}. Similarity: {similarity:.3f}")
                    print(f"     Question: {q_data['question']}")
                    print(f"     Answer: {q_data['answer'][:100]}...")
                    print()
        else:
            print(f"\n⚡ Step 2: Skipped vector search (Category: {classification_result['category']})")
        
        # Get final routing result
        print(f"\n🎯 Final Routing Result:")
        result = await self.route_query(query)
        
        print(f"   Route: {result['route']}")
        if result['route'] == "VECTOR_BASED":
            print(f"   ✅ Matched answer: {result['answer'][:150]}...")
            print(f"   📊 Similarity: {result['similarity_score']:.3f}")
        else:
            print(f"   ❌ Reason: {result['reason']}")
        
        return result
