from typing import Dict, Optional, List
from uuid import uuid4
import logging
import pandas as pd
import re
from rapidfuzz import fuzz
import os
import asyncio

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RuleBasedChatbot:
    def __init__(self):
        # Đọc dữ liệu patterns từ file Excel
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'data_test.xlsx')

        try:
            df = pd.read_excel(file_path)
            logger.info(f"Successfully loaded Excel file from: {file_path}")
        except FileNotFoundError:
            logger.error(f"Excel file not found at: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            raise

        self.patterns_list = []
        for idx, row in df.iterrows():
            patterns = [p.strip() for p in str(row['question']).split('|')]
            for pattern in patterns:
                self.patterns_list.append({
                    'pattern': pattern,
                    'answer': row['answer'],
                    'keywords': row.get('keywords', None)
                })

        logger.info(f"Loaded {len(self.patterns_list)} patterns for rule-based matching")

    async def preprocess(self, text):
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    async def exact_pattern_matching(self, user_input):
        user_input_p = await self.preprocess(user_input)
        for p in self.patterns_list:
            pattern = p['pattern']
            if pattern == '(.*)':
                continue
            if re.fullmatch(pattern, user_input_p):
                return p['answer']
        return None

    async def fuzzy_matching(self, user_input, threshold=80):
        user_input_p = await self.preprocess(user_input)
        best_score = 0
        best_answer = None
        for p in self.patterns_list:
            pattern = p['pattern']
            if pattern == '(.*)':
                continue
            pattern_processed = await self.preprocess(pattern)
            score = fuzz.token_set_ratio(user_input_p, pattern_processed)
            if score > best_score:
                best_score = score
                best_answer = p['answer']
        if best_score >= threshold:
            return best_answer
        return None

    async def keyword_probability_matching(self, user_input, threshold=0.5):
        user_input_p = await self.preprocess(user_input)
        user_words = set(user_input_p.split())
        best_score = 0
        best_answer = None
        for p in self.patterns_list:
            pattern_processed = await self.preprocess(p['pattern'])
            pattern_words = set(pattern_processed.split())
            if not pattern_words:
                continue
            match_count = len(user_words & pattern_words)
            prob = match_count / len(pattern_words) if pattern_words else 0
            if prob > best_score:
                best_score = prob
                best_answer = p['answer']
        if best_score >= threshold:
            return best_answer
        return None

    async def fallback_response(self):
        for p in self.patterns_list:
            if p['pattern'] == '(.*)':
                return p['answer']
        return "Xin lỗi, tôi chưa hiểu ý bạn. Bạn có thể diễn đạt lại không?"

    async def is_meaningful(self, text):
        text = text.strip()
        if re.fullmatch(r'[^a-zA-Zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ'
                        r'ìíịỉĩòóọỏõôồốộổỗơờớợởỡ'
                        r'ùúụủũưừứựửữỳýỵỷỹđ\s]+', text):
            return False
        if text.isdigit():
            return False
        if len(set(text)) == 1:
            return False
        return len(re.findall(r'[a-zA-Zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ'
                              r'ìíịỉĩòóọỏõôồốộổỗơờớợởỡ'
                              r'ùúụủũưừứựửữỳýỵỷỹđ]', text)) >= 3

    async def chatbot_response(self, user_input):
        if not await self.is_meaningful(user_input):
            return await self.fallback_response()

        answer = await self.exact_pattern_matching(user_input)
        if answer:
            return answer
        answer = await self.fuzzy_matching(user_input)
        if answer:
            return answer
        answer = await self.keyword_probability_matching(user_input)
        if answer:
            return answer
        return await self.fallback_response()
