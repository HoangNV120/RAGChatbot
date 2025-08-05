from typing import Dict, Optional, List, Tuple
import logging
from langchain_openai import OpenAIEmbeddings
from app.config import settings
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import asyncio

logger = logging.getLogger(__name__)

class VectorRouter:
    def __init__(self):
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model="text-embedding-ada-002"
        )
        
        # Load data từ data_test.xlsx
        self.data = pd.read_excel('app/data_test.xlsx')
        
        # Similarity threshold để quyết định có match không
        self.similarity_threshold = 0.75  # Có thể điều chỉnh
        
        # Cache cho embeddings
        self.question_embeddings = None
        self.questions_data = []
        
        print(f"VectorRouter initialized with {len(self.data)} questions")
        
    async def _initialize_embeddings(self):
        """Khởi tạo embeddings cho tất cả câu hỏi trong database"""
        
        if self.question_embeddings is not None:
            return  # Đã khởi tạo rồi
            
        print("🔄 Initializing question embeddings...")
        
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
    
    async def route_query(self, query: str) -> Dict:
        """Main routing function với vector similarity"""
        try:
            # Bước 1: Khởi tạo embeddings nếu chưa có
            await self._initialize_embeddings()
            
            print(f"🔍 Vector routing for query: {query[:50]}...")
            
            # Bước 2: Lấy embedding cho câu hỏi input
            query_embedding = await self._get_query_embedding(query)
            
            # Bước 3: Tìm câu hỏi tương tự nhất
            similar_questions = self._find_most_similar_question(query_embedding, top_k=3)
            
            if not similar_questions:
                print("❌ No similar questions found")
                return {
                    "route": "RAG_CHAT",
                    "reason": "No questions in database",
                    "query": query,
                    "similarity_score": 0.0
                }
            
            # Lấy câu hỏi có similarity cao nhất
            best_idx, best_similarity, best_question_data = similar_questions[0]
            
            print(f"📊 Best similarity: {best_similarity:.3f}")
            print(f"🔍 Best match: {best_question_data['question'][:50]}...")
            
            # Bước 4: Quyết định dựa trên threshold
            if best_similarity >= self.similarity_threshold:
                print(f"✅ Found good match (similarity: {best_similarity:.3f} >= {self.similarity_threshold})")
                
                return {
                    "route": "VECTOR_BASED",
                    "similarity_score": best_similarity,
                    "matched_question": best_question_data['question'],
                    "answer": best_question_data['answer'],
                    "source": best_question_data['source'],
                    "query": query,
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
                print(f"❌ No good match found (best similarity: {best_similarity:.3f} < {self.similarity_threshold})")
                
                return {
                    "route": "RAG_CHAT",
                    "reason": f"Best similarity {best_similarity:.3f} below threshold {self.similarity_threshold}",
                    "query": query,
                    "similarity_score": best_similarity,
                    "best_match": {
                        "question": best_question_data['question'],
                        "similarity": best_similarity,
                        "answer": best_question_data['answer'][:100] + "..." if len(best_question_data['answer']) > 100 else best_question_data['answer']
                    }
                }
            
        except Exception as e:
            logger.error(f"Error in vector routing: {e}")
            print(f"❌ Error in vector routing: {e}")
            return {
                "route": "RAG_CHAT",
                "reason": f"Error during vector routing: {str(e)}",
                "query": query,
                "similarity_score": 0.0
            }
    
    def get_stats(self) -> Dict:
        """Thống kê về vector router"""
        stats = {
            "total_questions": len(self.questions_data),
            "embedding_dimension": self.question_embeddings.shape[1] if self.question_embeddings is not None else 0,
            "similarity_threshold": self.similarity_threshold,
            "status": "ready" if self.question_embeddings is not None else "not_initialized"
        }
        
        return stats
    
    def set_similarity_threshold(self, threshold: float):
        """Điều chỉnh threshold"""
        if 0.0 <= threshold <= 1.0:
            self.similarity_threshold = threshold
            print(f"📊 Similarity threshold updated to: {threshold}")
        else:
            print(f"❌ Invalid threshold: {threshold}. Must be between 0.0 and 1.0")
    
    async def test_query(self, query: str, show_details: bool = True) -> Dict:
        """Test một câu hỏi và hiển thị chi tiết"""
        
        await self._initialize_embeddings()
        
        print(f"\n🧪 Testing query: '{query}'")
        
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
        
        # Get routing result
        result = await self.route_query(query)
        
        print(f"🎯 Routing result: {result['route']}")
        if result['route'] == "VECTOR_BASED":
            print(f"✅ Matched answer: {result['answer'][:150]}...")
        else:
            print(f"❌ Reason: {result['reason']}")
        
        return result
