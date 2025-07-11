from typing import Dict, Optional, List
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.config import settings
import pandas as pd
import asyncio

logger = logging.getLogger(__name__)

class CategoryRouter:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.model_name,
            temperature=0.1,
            api_key=settings.openai_api_key
        )
        
        # Load data để tạo categories
        self.data = pd.read_excel('app/data_test.xlsx')
        
        # Định nghĩa các categories chính dựa trên phân tích dữ liệu
        self.categories = {
            "NGÀNH_HỌC": {
                "description": "Câu hỏi về các ngành học, chương trình đào tạo, môn học tại FPT",
                "keywords": ["ngành", "chương trình", "đào tạo", "môn học", "khoa", "chuyên ngành"]
            },
            "QUY_CHE_THI": {
                "description": "Câu hỏi về quy chế thi, nội quy phòng thi, vi phạm khi thi",
                "keywords": ["thi", "phòng thi", "quy chế", "vi phạm", "kỷ luật", "nội quy"]
            },
            "ĐIỂM_SỐ_HỌC_TẬP": {
                "description": "Câu hỏi về điểm số, học tập, đánh giá, thang điểm",
                "keywords": ["điểm", "học tập", "đánh giá", "thang điểm", "kết quả"]
            },
            "HỌC_PHÍ_TÀI_CHÍNH": {
                "description": "Câu hỏi về học phí, tài chính, chi phí, học bổng",
                "keywords": ["học phí", "tiền", "chi phí", "học bổng", "tài chính"]
            },
            "SINH_VIÊN_DỊCH_VỤ": {
                "description": "Câu hỏi về các dịch vụ sinh viên, hỗ trợ, thủ tục hành chính",
                "keywords": ["sinh viên", "dịch vụ", "hỗ trợ", "thủ tục", "đăng ký"]
            },
            "CƠ_SỞ_VẬT_CHẤT": {
                "description": "Câu hỏi về cơ sở vật chất, phòng học, trang thiết bị",
                "keywords": ["phòng", "tòa", "cơ sở", "thiết bị", "địa điểm"]
            },
            "KHÁC": {
                "description": "Các câu hỏi khác không thuộc các category trên",
                "keywords": []
            }
        }
        
        # Category classification prompt
        self.category_prompt = """Bạn là một chuyên gia phân loại câu hỏi của sinh viên FPT.

Dựa trên câu hỏi được cung cấp, hãy xác định category phù hợp nhất từ các category sau:

1. NGÀNH_HỌC: Câu hỏi về các ngành học, chương trình đào tạo, môn học tại FPT
2. QUY_CHE_THI: Câu hỏi về quy chế thi, nội quy phòng thi, vi phạm khi thi  
3. ĐIỂM_SỐ_HỌC_TẬP: Câu hỏi về điểm số, học tập, đánh giá, thang điểm
4. HỌC_PHÍ_TÀI_CHÍNH: Câu hỏi về học phí, tài chính, chi phí, học bổng
5. SINH_VIÊN_DỊCH_VỤ: Câu hỏi về các dịch vụ sinh viên, hỗ trợ, thủ tục hành chính
6. CƠ_SỞ_VẬT_CHẤT: Câu hỏi về cơ sở vật chất, phòng học, trang thiết bị
7. KHÁC: Các câu hỏi khác không thuộc các category trên

Câu hỏi: "{query}"

Chỉ trả về tên category (ví dụ: NGÀNH_HỌC), không giải thích thêm."""

        # Tạo mapping question -> category cho database
        self.question_categories = self._create_question_categories()
        
    def _create_question_categories(self) -> Dict[str, str]:
        """Tạo mapping question -> category cho database"""
        mapping = {}
        
        for idx, row in self.data.iterrows():
            question = str(row['question']).lower()
            
            # Rule-based categorization dựa trên keywords
            category = "KHÁC"  # default
            
            if any(keyword in question for keyword in ["ngành", "chương trình", "đào tạo", "môn học", "khoa"]):
                category = "NGÀNH_HỌC"
            elif any(keyword in question for keyword in ["thi", "phòng thi", "quy chế", "vi phạm", "kỷ luật"]):
                category = "QUY_CHE_THI"
            elif any(keyword in question for keyword in ["điểm", "học tập", "đánh giá", "thang điểm"]):
                category = "ĐIỂM_SỐ_HỌC_TẬP"
            elif any(keyword in question for keyword in ["học phí", "tiền", "chi phí", "học bổng"]):
                category = "HỌC_PHÍ_TÀI_CHÍNH"
            elif any(keyword in question for keyword in ["sinh viên", "dịch vụ", "hỗ trợ", "thủ tục", "đăng ký"]):
                category = "SINH_VIÊN_DỊCH_VỤ"
            elif any(keyword in question for keyword in ["phòng", "tòa", "cơ sở", "thiết bị", "địa điểm"]):
                category = "CƠ_SỞ_VẬT_CHẤT"
                
            mapping[question] = category
            
        return mapping
    
    async def classify_query_category(self, query: str) -> str:
        """Sử dụng LLM để phân loại category của câu hỏi"""
        try:
            prompt = self.category_prompt.format(query=query)
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            
            category = response.content.strip().upper()
            
            # Validate category
            if category in self.categories:
                return category
            else:
                logger.warning(f"Invalid category returned: {category}, defaulting to KHÁC")
                return "KHÁC"
                
        except Exception as e:
            logger.error(f"Error in category classification: {e}")
            return "KHÁC"
    
    def get_questions_by_category(self, category: str) -> List[Dict]:
        """Lấy tất cả câu hỏi thuộc một category"""
        questions = []
        
        for idx, row in self.data.iterrows():
            question_lower = str(row['question']).lower()
            if self.question_categories.get(question_lower, "KHÁC") == category:
                questions.append({
                    "question": row['question'],
                    "answer": row['answer'],
                    "source": row.get('nguồn', ''),
                    "index": idx
                })
                
        return questions
    
    async def find_best_match_in_category(self, query: str, category: str) -> Optional[Dict]:
        """Tìm câu trả lời tốt nhất trong một category"""
        
        # Lấy tất cả câu hỏi trong category
        category_questions = self.get_questions_by_category(category)
        
        if not category_questions:
            return None
            
        # Sử dụng LLM để tìm câu hỏi tương tự nhất
        find_match_prompt = f"""Bạn là một chuyên gia tìm kiếm câu hỏi tương tự.

Câu hỏi cần tìm: "{query}"

Danh sách câu hỏi trong category {category}:
"""
        
        for i, q in enumerate(category_questions[:10]):  # Giới hạn 10 câu để tránh prompt quá dài
            find_match_prompt += f"{i+1}. {q['question']}\n"
            
        find_match_prompt += f"""
Hãy chọn câu hỏi tương tự nhất với câu hỏi cần tìm.
Chỉ trả về số thứ tự (ví dụ: 3), không giải thích thêm.
Nếu không có câu nào tương tự, trả về 0.
"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=find_match_prompt)])
            match_index = int(response.content.strip())
            
            if 1 <= match_index <= len(category_questions[:10]):
                return category_questions[match_index - 1]
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error finding match in category: {e}")
            return None
    
    async def route_query(self, query: str) -> Dict:
        """Main routing function"""
        try:
            # Bước 1: Classify category
            category = await self.classify_query_category(query)
            
            logger.info(f"Query classified as category: {category}")
            print(f"🔍 Category detected: {category}")
            
            # Bước 2: Nếu category là KHÁC, tự động đi RAG_CHAT
            if category == "KHÁC":
                print(f"⚡ Category is KHÁC, automatically routing to RAG_CHAT")
                return {
                    "route": "RAG_CHAT",
                    "category": category,
                    "reason": "Category classified as KHÁC - using RAG for better response",
                    "query": query  # Original query
                }
            
            # Bước 3: Tìm match trong category cụ thể
            match = await self.find_best_match_in_category(query, category)
            
            if match:
                print(f"✅ Found match in category {category}")
                return {
                    "route": "CATEGORY_BASED",
                    "category": category,
                    "matched_question": match['question'],
                    "answer": match['answer'],
                    "source": match['source'],
                    "query": query  # Original query
                }
            
            # Bước 4: Nếu không tìm thấy trong category, fallback to RAG
            print(f"❌ No match found in category {category}, falling back to RAG")
            return {
                "route": "RAG_CHAT",
                "category": category,
                "reason": f"No suitable match found in category {category}",
                "query": query  # Original query
            }
            
        except Exception as e:
            logger.error(f"Error in category routing: {e}")
            return {
                "route": "RAG_CHAT",
                "category": "ERROR",
                "reason": f"Error during routing: {str(e)}",
                "query": query
            }
    
    def get_category_stats(self) -> Dict:
        """Thống kê categories"""
        stats = {}
        total_questions = len(self.data)
        
        for category in self.categories.keys():
            count = sum(1 for cat in self.question_categories.values() if cat == category)
            stats[category] = {
                "count": count,
                "percentage": (count / total_questions) * 100
            }
            
        return stats
