"""
Advanced Rule-Based Chatbot - Top-K Semantic + Normalized Scoring
"""

import pandas as pd
import re
import asyncio
from typing import Optional, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import unicodedata

class AdvancedChatbot:
    def __init__(self, data_file: str = 'app/data.xlsx'):
        """Khởi tạo chatbot nâng cao với top-k semantic"""
        print("Loading data...")
        self.data = pd.read_excel(data_file)
        self.data = self.data.dropna(subset=['question', 'answer'])
        self.data['question'] = self.data['question'].astype(str)
        self.data['answer'] = self.data['answer'].astype(str)

        # Preprocessing nâng cao
        self.data['processed_question'] = self.data['question'].apply(self.preprocess_text)
        self.data['normalized_question'] = self.data['question'].apply(self.normalize_question)

        # Sử dụng keyword có sẵn
        if 'keyword' in self.data.columns:
            self.data['keywords'] = self.data['keyword'].astype(str).fillna('')
        else:
            self.data['keywords'] = ''

        # Tạo multiple TF-IDF matrices cho different strategies
        self._create_tfidf_matrices()

        print(f"Loaded {len(self.data)} questions successfully!")

    def _create_tfidf_matrices(self):
        """Tạo nhiều TF-IDF matrix cho các chiến lược khác nhau"""

        # Matrix 1: Processed questions + keywords
        combined_text1 = []
        for _, row in self.data.iterrows():
            text = f"{row['processed_question']} {row['keywords']}"
            combined_text1.append(text)

        self.tfidf_vectorizer1 = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),  # Tăng lên 3-gram
            min_df=1,
            max_df=0.7,
            lowercase=True
        )
        self.tfidf_matrix1 = self.tfidf_vectorizer1.fit_transform(combined_text1)

        # Matrix 2: Normalized questions only
        self.tfidf_vectorizer2 = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8,
            lowercase=True
        )
        self.tfidf_matrix2 = self.tfidf_vectorizer2.fit_transform(self.data['normalized_question'].tolist())

        # Matrix 3: Original questions
        self.tfidf_vectorizer3 = TfidfVectorizer(
            max_features=6000,
            ngram_range=(2, 4),  # Focus on phrases
            min_df=1,
            max_df=0.9,
            lowercase=True
        )
        self.tfidf_matrix3 = self.tfidf_vectorizer3.fit_transform(self.data['question'].str.lower().tolist())

    def normalize_question(self, text: str) -> str:
        """Enhanced normalize câu hỏi để tăng khả năng match"""
        text = self.preprocess_text(text)

        # Chuẩn hóa câu hỏi patterns
        question_normalizations = {
            # Question words normalization
            r'\b(những|các)\s+': '',
            r'\b(tại|ở)\s+(fptu?|đại học fpt|trường fpt)\b': 'fpt',
            r'\b(fptu?|đại học fpt|trường fpt)\s+(có|tại)\b': 'fpt',
            r'\b(bao nhiêu|mấy)\s+(phút|giờ|thời gian)\b': 'bao nhiêu thời gian',
            r'\b(bao nhiêu|mấy)\s+(tiền|phí)\b': 'bao nhiêu phí',
            r'\b(nào|gì)\s*$': '',  # Remove trailing question words
            r'\b(như thế nào|làm sao|cách nào|ra sao)\b': 'như thế nào',
            r'\b(ở đâu|tại đâu|chỗ nào|nơi nào)\b': 'ở đâu',
            r'\b(khi nào|lúc nào|thời gian nào)\b': 'khi nào',

            # Content normalization
            r'\b(sinh viên|học sinh|sv)\b': 'sinh viên',
            r'\b(học phí|chi phí học|phí học)\b': 'học phí',
            r'\b(ngành học|chuyên ngành|ngành đào tạo)\b': 'ngành học',
            r'\b(phòng thi|phòng kiểm tra)\b': 'phòng thi',
            r'\b(điểm thi|kết quả thi)\b': 'điểm thi',
            r'\b(đại học fpt|trường fpt|fpt university)\b': 'fpt',

            # Time expressions
            r'\b(thời gian|giờ giấc|lịch trình)\b': 'thời gian',
            r'\b(phút|giờ)\b': 'thời gian',

            # Common phrases
            r'\b(có được|được không|có thể)\b': 'được',
            r'\b(cần phải|phải|cần)\b': 'cần',
            r'\b(và|với|cùng)\b': 'và',
        }

        # Apply normalizations
        for pattern, replacement in question_normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Remove redundant words
        redundant_words = ['thì', 'mà', 'rồi', 'đây', 'kia', 'này', 'đó', 'vậy', 'ạ', 'ơi']
        words = text.split()
        words = [word for word in words if word not in redundant_words]
        text = ' '.join(words)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def preprocess_text(self, text: str) -> str:
        """Enhanced preprocessing với better synonym handling"""
        if not text:
            return ""

        text = str(text).lower().strip()
        text = unicodedata.normalize('NFC', text)

        # Enhanced synonym and abbreviation handling
        text_normalizations = {
            # Abbreviations
            r'\b(sv|sinh viên)\b': 'sinh viên',
            r'\b(fptu?|đại học fpt|trường fpt|fpt university)\b': 'fpt',
            r'\b(đh|đại học)\b': 'đại học',
            r'\b(bn|bao nhiêu)\b': 'bao nhiêu',
            r'\b(k|ko|không)\b': 'không',
            r'\b(đc|được)\b': 'được',
            r'\b(vs|với)\b': 'với',
            r'\b(pt|bài kiểm tra|kiểm tra quá trình)\b': 'bài kiểm tra',
            r'\b(mấy|bao nhiêu)\b': 'bao nhiêu',
            r'\b(tg|thời gian)\b': 'thời gian',
            r'\b(hp|học phí)\b': 'học phí',
            r'\b(hs|học sinh)\b': 'sinh viên',
            r'\b(ojt|OJT)\b': 'thực tập',
            r'\b(fe|FE|Điểm cuối kì|Bài thi cuối kì)\b': 'Bài thi cuối kì',
            r'\b(Midterm|ME|Điểm giữa kì|Bài thi giữa kì)\b': 'Bài thi giữa kì',

            # Wildcard patterns - Generic question structures
            r'\b(có|tồn tại|hiện có)\s+(bao nhiêu|mấy|những|các)\b': 'có bao nhiêu',
            r'\b(cần|phải|yêu cầu)\s+(bao nhiêu|mấy|những gì)\b': 'cần bao nhiêu',
            r'\b(thời gian|thời hạn|deadline)\s+(bao lâu|như thế nào|ra sao)\b': 'thời gian bao lâu',
            r'\b(điều kiện|yêu cầu|tiêu chuẩn)\s+(gì|nào|như thế nào)\b': 'điều kiện gì',
            r'\b(quy định|luật|chính sách)\s+(về|đối với|cho)\b': 'quy định về',
            r'\b(cách thức|phương pháp|hướng dẫn)\s+(để|cho|nhằm)\b': 'cách thức để',

            # Question words standardization
            r'\b(như thế nào|làm sao|cách nào|ra sao)\b': 'như thế nào',
            r'\b(ở đâu|tại đâu|chỗ nào|nơi nào)\b': 'ở đâu',
            r'\b(khi nào|lúc nào|thời gian nào)\b': 'khi nào',
            r'\b(bao lâu|mất bao lâu)\b': 'bao lâu',
            r'\b(có được|được không|có thể)\b': 'được',

            # Content synonyms
            r'\b(chi phí|phí tổn|giá cả)\b': 'phí',
            r'\b(chuyên ngành|ngành đào tạo)\b': 'ngành',
            r'\b(kiểm tra|bài thi)\b': 'thi',
            r'\b(học sinh|sinh viên)\b': 'sinh viên',
            r'\b(giáo viên|thầy cô|giảng viên)\b': 'giảng viên',
            r'\b(phòng học|lớp học)\b': 'phòng',
            r'\b(thời khóa biểu|lịch học)\b': 'lịch',

            # Units and measurements
            r'\b(triệu|tr)\b': 'triệu',
            r'\b(nghìn|k)\b': 'nghìn',
            r'\b(giờ|h)\b': 'giờ',
            r'\b(phút|ph)\b': 'phút',
        }

        # Apply normalizations
        for pattern, replacement in text_normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Clean punctuation and extra spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    async def top_k_semantic_search(self, user_input: str, k: int = 8) -> List[Tuple[int, float]]:
        """Top-K semantic search với multiple strategies - tăng k để có nhiều candidate hơn"""
        await asyncio.sleep(0)  # Yield control to event loop
        candidates = []

        user_processed = self.preprocess_text(user_input)
        user_normalized = self.normalize_question(user_input)

        # Strategy 1: TF-IDF Matrix 1 (processed + keywords)
        try:
            input_vector1 = self.tfidf_vectorizer1.transform([user_processed])
            similarities1 = cosine_similarity(input_vector1, self.tfidf_matrix1).flatten()
            top_indices1 = np.argsort(similarities1)[-k*2:][::-1]  # Lấy nhiều hơn để rerank
            for idx in top_indices1:
                if similarities1[idx] > 0.08:  # Giảm threshold
                    candidates.append((idx, similarities1[idx] * 1.0, 'tfidf1'))
        except:
            pass

        # Strategy 2: TF-IDF Matrix 2 (normalized)
        try:
            input_vector2 = self.tfidf_vectorizer2.transform([user_normalized])
            similarities2 = cosine_similarity(input_vector2, self.tfidf_matrix2).flatten()
            top_indices2 = np.argsort(similarities2)[-k*2:][::-1]
            for idx in top_indices2:
                if similarities2[idx] > 0.08:
                    candidates.append((idx, similarities2[idx] * 1.2, 'tfidf2'))
        except:
            pass

        # Strategy 3: TF-IDF Matrix 3 (phrases)
        try:
            input_vector3 = self.tfidf_vectorizer3.transform([user_input.lower()])
            similarities3 = cosine_similarity(input_vector3, self.tfidf_matrix3).flatten()
            top_indices3 = np.argsort(similarities3)[-k*2:][::-1]
            for idx in top_indices3:
                if similarities3[idx] > 0.04:
                    candidates.append((idx, similarities3[idx] * 0.8, 'tfidf3'))
        except:
            pass

        # Normalize scores và aggregate
        if candidates:
            # Group by index
            score_dict = {}
            for idx, score, strategy in candidates:
                if idx not in score_dict:
                    score_dict[idx] = []
                score_dict[idx].append(score)

            # Calculate final scores
            final_candidates = []
            for idx, scores in score_dict.items():
                # Use max + average boost + strategy diversity bonus
                final_score = max(scores) + (sum(scores) / len(scores)) * 0.3
                if len(scores) > 1:  # Bonus for multiple strategies
                    final_score += 0.1
                final_candidates.append((idx, final_score))

            # Sort and return top-k
            final_candidates.sort(key=lambda x: x[1], reverse=True)
            return final_candidates[:k]

        return []

    async def exact_match(self, user_input: str) -> Optional[str]:
        """Exact matching với nhiều variants"""
        await asyncio.sleep(0)  # Yield control to event loop
        variants = [
            self.preprocess_text(user_input),
            self.normalize_question(user_input),
            user_input.lower().strip()
        ]

        for variant in variants:
            for _, row in self.data.iterrows():
                question_variants = [
                    row['processed_question'],
                    row['normalized_question'],
                    row['question'].lower().strip()
                ]
                if variant in question_variants:
                    return row['answer']
        return None

    async def advanced_keyword_match(self, user_input: str) -> Optional[str]:
        """Advanced keyword matching với context-aware scoring"""
        await asyncio.sleep(0)  # Yield control to event loop
        user_processed = self.preprocess_text(user_input)
        user_normalized = self.normalize_question(user_input)
        user_words = set(user_processed.split())
        user_norm_words = set(user_normalized.split())

        # Context-aware important words với dynamic weights
        context_weights = {
            'ngành': {'học': 1.5, 'fpt': 1.3, 'đại': 1.2, 'trường': 1.2},
            'học': {'phí': 1.5, 'ngành': 1.3, 'môn': 1.2, 'cách': 1.2},
            'phí': {'học': 1.5, 'bao': 1.4, 'nhiêu': 1.4, 'tiền': 1.3},
            'thi': {'phòng': 1.5, 'điểm': 1.3, 'trễ': 1.4, 'cấm': 1.3},
            'sinh': {'viên': 1.8, 'học': 1.2, 'trường': 1.2},
            'fpt': {'ngành': 1.3, 'học': 1.2, 'đại': 1.2, 'trường': 1.2}
        }

        # Base important words
        important_words = {
            'ngành': 18, 'học': 15, 'phí': 18, 'điểm': 15, 'thi': 15, 'phòng': 12,
            'sinh': 10, 'viên': 10, 'fpt': 15, 'môn': 12, 'bao': 10, 'nhiêu': 10,
            'mấy': 10, 'khi': 8, 'ở': 8, 'đâu': 8, 'nào': 6, 'gì': 6,
            'đại': 10, 'trường': 8, 'thời': 8, 'gian': 8, 'cách': 8, 'làm': 6
        }

        scored_matches = []

        for idx, row in self.data.iterrows():
            question_words = set(row['processed_question'].split())
            norm_question_words = set(row['normalized_question'].split())

            score = 0

            # === CONTEXT-AWARE MATCHING ===
            # Check for context combinations
            context_bonus = 0
            for main_word in user_words:
                if main_word in context_weights:
                    for context_word, multiplier in context_weights[main_word].items():
                        if context_word in user_words and main_word in question_words and context_word in question_words:
                            context_bonus += important_words.get(main_word, 5) * multiplier

            score += context_bonus

            # === EXACT MATCHES ===
            exact_matches = user_words.intersection(question_words)
            norm_matches = user_norm_words.intersection(norm_question_words)

            # Score exact matches với context awareness
            for word in exact_matches:
                base_score = important_words.get(word, 4)
                # Boost if word appears in context
                if word in context_weights:
                    context_found = any(cw in user_words for cw in context_weights[word].keys())
                    if context_found:
                        base_score *= 1.3
                score += base_score

            # Score normalized matches
            for word in norm_matches:
                if word not in exact_matches:  # Avoid double counting
                    score += important_words.get(word, 3) * 0.7

            # === COVERAGE SCORING ===
            if len(user_words) > 0:
                coverage = len(exact_matches) / len(user_words)
                score += coverage * 25

                # Bonus for high coverage
                if coverage > 0.7:
                    score += 15

            if len(user_norm_words) > 0:
                norm_coverage = len(norm_matches) / len(user_norm_words)
                score += norm_coverage * 20

            # === KEYWORD COLUMN MATCHING ===
            if row['keywords']:
                keyword_words = set(str(row['keywords']).lower().split())
                keyword_matches = user_words.intersection(keyword_words)

                # Context-aware keyword scoring
                for word in keyword_matches:
                    base_score = important_words.get(word, 8) * 2
                    if word in context_weights:
                        context_found = any(cw in user_words for cw in context_weights[word].keys())
                        if context_found:
                            base_score *= 1.4
                    score += base_score

            # === PARTIAL MATCHING ===
            partial_score = 0
            for user_word in user_words:
                if len(user_word) > 3:
                    for q_word in question_words:
                        if len(q_word) > 3:
                            if user_word in q_word or q_word in user_word:
                                partial_score += 8
                                break
            score += partial_score

            # === QUESTION TYPE MATCHING ===
            # Identify question patterns
            question_patterns = {
                'what': ['gì', 'nào', 'những', 'các'],
                'how_many': ['bao', 'nhiêu', 'mấy'],
                'when': ['khi', 'lúc', 'thời'],
                'where': ['ở', 'đâu', 'chỗ'],
                'how': ['như', 'thế', 'nào', 'cách', 'làm']
            }

            user_pattern = None
            row_pattern = None

            for pattern, words in question_patterns.items():
                if any(word in user_words for word in words):
                    user_pattern = pattern
                if any(word in question_words for word in words):
                    row_pattern = pattern

            if user_pattern and user_pattern == row_pattern:
                score += 15

            # === LENGTH AND STRUCTURE SIMILARITY ===
            len_ratio = min(len(user_input), len(row['question'])) / max(len(user_input), len(row['question']))
            score += len_ratio * 8

            # Word count similarity
            word_count_ratio = min(len(user_words), len(question_words)) / max(len(user_words), len(question_words))
            score += word_count_ratio * 5

            if score > 20:  # Threshold
                scored_matches.append((score, row['answer'], idx))

        # Normalize scores và return best match
        if scored_matches:
            scored_matches.sort(key=lambda x: x[0], reverse=True)

            # Dynamic threshold based on score distribution
            top_score = scored_matches[0][0]
            if len(scored_matches) > 1:
                second_score = scored_matches[1][0]
                threshold_ratio = 0.4 if top_score > second_score * 1.5 else 0.6
            else:
                threshold_ratio = 0.5

            max_score = max(scored_matches, key=lambda x: x[0])[0]
            if max_score > 0:
                normalized_score = scored_matches[0][0] / max_score
                if normalized_score > threshold_ratio:
                    return scored_matches[0][1]

        return None

    async def wildcard_match(self, user_input: str) -> Optional[str]:
        """Enhanced wildcard matching cho flexible pattern recognition"""
        await asyncio.sleep(0)  # Yield control to event loop
        user_processed = self.preprocess_text(user_input)
        user_normalized = self.normalize_question(user_input)
        user_words = user_processed.split()

        # Wildcard patterns với scoring
        wildcard_patterns = [
            # Educational patterns
            {
                'pattern': r'\b(ngành|chuyên ngành|học|đào tạo).*(có|tồn tại|hiện có|bao nhiêu)',
                'keywords': ['ngành', 'học', 'có', 'bao', 'nhiêu'],
                'weight': 25,
                'category': 'education'
            },
            {
                'pattern': r'\b(học phí|chi phí|phí|tiền).*(bao nhiêu|mấy|giá)',
                'keywords': ['học', 'phí', 'bao', 'nhiêu', 'tiền'],
                'weight': 25,
                'category': 'cost'
            },
            {
                'pattern': r'\b(thi|kiểm tra|bài thi).*(phòng|nơi|chỗ|ở đâu)',
                'keywords': ['thi', 'phòng', 'ở', 'đâu'],
                'weight': 20,
                'category': 'exam_location'
            },
            {
                'pattern': r'\b(điểm|kết quả|đánh giá).*(thi|kiểm tra|bài thi)',
                'keywords': ['điểm', 'thi', 'kết quả'],
                'weight': 20,
                'category': 'exam_result'
            },
            {
                'pattern': r'\b(thời gian|giờ|lúc|khi).*(thi|kiểm tra|học|bắt đầu)',
                'keywords': ['thời', 'gian', 'thi', 'khi'],
                'weight': 18,
                'category': 'time'
            },
            {
                'pattern': r'\b(sinh viên|học sinh|sv).*(được|có thể|phải|cần)',
                'keywords': ['sinh', 'viên', 'được', 'phải'],
                'weight': 15,
                'category': 'student_rights'
            },
            {
                'pattern': r'\b(quy định|luật|chính sách|điều kiện).*(về|đối với|cho)',
                'keywords': ['quy', 'định', 'về', 'điều', 'kiện'],
                'weight': 15,
                'category': 'rules'
            },
            {
                'pattern': r'\b(cách|phương pháp|hướng dẫn).*(để|cho|nhằm|thực hiện)',
                'keywords': ['cách', 'để', 'thực', 'hiện'],
                'weight': 12,
                'category': 'procedure'
            },
            # Advanced patterns
            {
                'pattern': r'\b(trễ|muộn|chậm).*(bao nhiêu|mấy|thời gian)',
                'keywords': ['trễ', 'bao', 'nhiêu', 'thời', 'gian'],
                'weight': 22,
                'category': 'late_policy'
            },
            {
                'pattern': r'\b(cấm|không được|bị cấm).*(gì|nào|hành vi)',
                'keywords': ['cấm', 'không', 'được', 'gì', 'hành', 'vi'],
                'weight': 20,
                'category': 'prohibited'
            },
            {
                'pattern': r'\b(fpt|đại học|trường).*(có|tồn tại|hiện có)',
                'keywords': ['fpt', 'đại', 'học', 'có'],
                'weight': 18,
                'category': 'university'
            },
            {
                'pattern': r'\b(thực tập|ojt|internship).*(như thế nào|cách|thời gian)',
                'keywords': ['thực', 'tập', 'như', 'thế', 'nào'],
                'weight': 20,
                'category': 'internship'
            }
        ]

        best_matches = []

        for idx, row in self.data.iterrows():
            question_processed = row['processed_question']
            question_normalized = row['normalized_question']
            question_words = set(question_processed.split())

            total_score = 0
            matched_patterns = []

            # Test against each wildcard pattern
            for pattern_info in wildcard_patterns:
                pattern = pattern_info['pattern']
                keywords = pattern_info['keywords']
                weight = pattern_info['weight']
                category = pattern_info['category']

                # Check if pattern matches user input
                user_match = re.search(pattern, user_processed, re.IGNORECASE)
                question_match = re.search(pattern, question_processed, re.IGNORECASE)

                if user_match or question_match:
                    # Score based on keyword overlap
                    keyword_overlap = 0
                    for keyword in keywords:
                        if keyword in user_processed and keyword in question_processed:
                            keyword_overlap += 1

                    if keyword_overlap > 0:
                        pattern_score = weight * (keyword_overlap / len(keywords))
                        total_score += pattern_score
                        matched_patterns.append((category, pattern_score))

            # Additional scoring for direct keyword matches
            user_word_set = set(user_processed.split())
            question_word_set = set(question_processed.split())

            # Important word boosting
            important_words = {
                'ngành': 15, 'học': 12, 'phí': 18, 'thi': 15, 'điểm': 15,
                'phòng': 12, 'sinh': 10, 'viên': 10, 'fpt': 15, 'thời': 10,
                'gian': 10, 'cách': 8, 'quy': 8, 'định': 8, 'thực': 12, 'tập': 12
            }

            for word in user_word_set.intersection(question_word_set):
                if word in important_words:
                    total_score += important_words[word]

            # Coverage bonus
            if len(user_word_set) > 0:
                coverage = len(user_word_set.intersection(question_word_set)) / len(user_word_set)
                total_score += coverage * 20

            # Pattern diversity bonus
            unique_categories = set(cat for cat, _ in matched_patterns)
            if len(unique_categories) > 1:
                total_score += 10

            # Keyword column matching
            if row['keywords']:
                keyword_words = set(str(row['keywords']).lower().split())
                keyword_matches = user_word_set.intersection(keyword_words)
                if keyword_matches:
                    total_score += len(keyword_matches) * 15

            if total_score > 25:  # Threshold for wildcard matching
                best_matches.append((total_score, row['answer'], idx, matched_patterns))

        # Return best match
        if best_matches:
            best_matches.sort(key=lambda x: x[0], reverse=True)
            best_match = best_matches[0]

            # Additional validation
            if len(best_matches) > 1:
                top_score = best_match[0]
                second_score = best_matches[1][0]

                # If top score is significantly better
                if top_score > second_score * 1.2:
                    return best_match[1]
                elif top_score > 40:  # High confidence threshold
                    return best_match[1]
            elif best_match[0] > 30:  # Single match threshold
                return best_match[1]

        return None

    async def hybrid_top_k_search(self, user_input: str) -> Optional[str]:
        """Hybrid Top-K search với fuzzy + keyword reranking"""
        await asyncio.sleep(0)  # Yield control to event loop
        candidates = await self.top_k_semantic_search(user_input, k=12)

        if not candidates:
            return None

        # Prepare user features
        user_processed = self.preprocess_text(user_input)
        user_normalized = self.normalize_question(user_input)
        user_words = set(user_processed.split())  # set for intersection operations
        user_words_list = user_processed.split()  # list for slicing
        user_norm_words = set(user_normalized.split())

        # Important words with higher weights
        important_words = {
            'ngành': 20, 'học': 15, 'phí': 20, 'điểm': 18, 'thi': 18, 'phòng': 15,
            'sinh': 12, 'viên': 12, 'fpt': 18, 'môn': 15, 'bao': 12, 'nhiêu': 12,
            'khi': 10, 'ở': 10, 'đâu': 10, 'nào': 8, 'gì': 8, 'như': 8,
            'đại': 12, 'trường': 10, 'thời': 10, 'gian': 10, 'cách': 8
        }

        # Re-rank candidates với advanced scoring
        reranked_candidates = []

        for idx, semantic_score in candidates:
            row = self.data.iloc[idx]

            # Initialize component scores
            fuzzy_score = 0
            keyword_score = 0
            context_score = 0

            # === FUZZY MATCHING COMPONENT ===
            question_words = set(row['processed_question'].split())
            norm_question_words = set(row['normalized_question'].split())

            if user_words and question_words:
                # Multiple similarity metrics
                intersection = len(user_words & question_words)
                union = len(user_words | question_words)
                jaccard = intersection / union if union > 0 else 0

                # Containment in both directions
                containment1 = intersection / len(user_words) if len(user_words) > 0 else 0
                containment2 = intersection / len(question_words) if len(question_words) > 0 else 0

                # Normalized word overlap
                norm_intersection = len(user_norm_words & norm_question_words)
                norm_union = len(user_norm_words | norm_question_words)
                norm_jaccard = norm_intersection / norm_union if norm_union > 0 else 0

                # Dice coefficient
                dice = 2 * intersection / (len(user_words) + len(question_words)) if (len(user_words) + len(question_words)) > 0 else 0

                # Combined fuzzy score
                fuzzy_score = (
                        jaccard * 0.35 +
                        containment1 * 0.25 +
                        containment2 * 0.15 +
                        norm_jaccard * 0.2 +
                        dice * 0.05
                )

            # === KEYWORD MATCHING COMPONENT ===
            # Important word matches
            important_matches = 0
            for word in user_words:
                if word in important_words:
                    if word in question_words:
                        important_matches += important_words[word]

            # Regular word matches
            regular_matches = len(user_words & question_words) * 5

            # Keyword column matches
            keyword_matches = 0
            if row['keywords']:
                keyword_words = set(str(row['keywords']).lower().split())
                keyword_overlap = len(user_words.intersection(keyword_words))
                keyword_matches = keyword_overlap * 25

            # Phrase matches
            phrase_matches = 0
            user_text = user_processed
            question_text = row['processed_question']

            # N-gram matches - use user_words_list for slicing
            for n in range(2, min(5, len(user_words_list) + 1)):
                user_ngrams = [' '.join(user_words_list[i:i+n]) for i in range(len(user_words_list) - n + 1)]
                for ngram in user_ngrams:
                    if ngram in question_text:
                        phrase_matches += n * 8

            # Substring matches
            if len(user_normalized) > 4 and user_normalized in row['normalized_question']:
                phrase_matches += 30

            keyword_score = (important_matches + regular_matches + keyword_matches + phrase_matches) / 100

            # === CONTEXT SCORING ===
            # Length similarity
            len_ratio = min(len(user_input), len(row['question'])) / max(len(user_input), len(row['question']))
            context_score += len_ratio * 0.3

            # Question type similarity
            user_question_words = {'gì', 'nào', 'bao', 'nhiêu', 'khi', 'ở', 'đâu', 'như', 'thế', 'sao'}
            user_has_question = any(word in user_words for word in user_question_words)
            row_has_question = any(word in question_words for word in user_question_words)

            if user_has_question and row_has_question:
                context_score += 0.2

            # === FINAL SCORING ===
            # Normalize semantic score (0-1)
            normalized_semantic = min(semantic_score, 1.0)

            # Weighted combination
            final_score = (
                    normalized_semantic * 0.4 +  # Semantic base
                    fuzzy_score * 0.35 +         # Fuzzy similarity
                    keyword_score * 0.2 +        # Keyword matching
                    context_score * 0.05         # Context bonus
            )

            # Boost for multiple high scores
            if fuzzy_score > 0.4 and keyword_score > 0.3:
                final_score *= 1.15

            if normalized_semantic > 0.3 and fuzzy_score > 0.3:
                final_score *= 1.1

            reranked_candidates.append((final_score, row['answer'], idx))

        # Return best candidate
        if reranked_candidates:
            reranked_candidates.sort(key=lambda x: x[0], reverse=True)
            best_candidate = reranked_candidates[0]

            # Dynamic threshold based on top candidates
            if len(reranked_candidates) > 1:
                top_score = best_candidate[0]
                second_score = reranked_candidates[1][0]

                # If top score is significantly better, lower threshold
                if top_score > second_score * 1.3:
                    threshold = 0.15
                else:
                    threshold = 0.25
            else:
                threshold = 0.2

            if best_candidate[0] > threshold:
                return best_candidate[1]

        return None

    async def ensemble_semantic_search(self, user_input: str) -> Optional[str]:
        """Ensemble semantic search với top-k - keep for backward compatibility"""
        return await self.hybrid_top_k_search(user_input)

    async def phrase_match(self, user_input: str) -> Optional[str]:
        """Enhanced phrase matching với context-aware scoring"""
        await asyncio.sleep(0)  # Yield control to event loop
        user_processed = self.preprocess_text(user_input)
        user_normalized = self.normalize_question(user_input)
        user_words = user_processed.split()

        # Important phrases với weights
        important_phrases = {
            'ngành học': 25, 'học phí': 25, 'phòng thi': 20, 'sinh viên': 15,
            'điểm thi': 20, 'bao nhiêu': 15, 'thời gian': 15, 'cách thức': 12,
            'như thế nào': 12, 'ở đâu': 12, 'khi nào': 12, 'đại học': 15,
            'trường fpt': 15, 'fpt university': 15, 'những gì': 10, 'làm sao': 10
        }

        best_match = None
        best_score = 0

        for idx, row in self.data.iterrows():
            question_processed = row['processed_question']
            question_normalized = row['normalized_question']
            question_words = question_processed.split()
            score = 0

            # === IMPORTANT PHRASE MATCHING ===
            for phrase, weight in important_phrases.items():
                if phrase in user_processed and phrase in question_processed:
                    score += weight
                elif phrase in user_normalized and phrase in question_normalized:
                    score += weight * 0.8

            # === MULTI-LEVEL N-GRAM MATCHING ===
            for n in range(2, min(7, len(user_words) + 1)):
                for i in range(len(user_words) - n + 1):
                    ngram = ' '.join(user_words[i:i+n])

                    # Skip if ngram is too short or common
                    if len(ngram) < 4:
                        continue

                    # Score based on n-gram length và rarity
                    if ngram in question_processed:
                        # Bonus for longer and rarer n-grams
                        rarity_bonus = 1.0
                        if n >= 4:
                            rarity_bonus = 1.5
                        if n >= 5:
                            rarity_bonus = 2.0

                        score += n * 12 * rarity_bonus

                    if ngram in question_normalized:
                        score += n * 10

            # === SUBSTRING MATCHING ===
            # Long substring matches
            if len(user_normalized) > 5:
                if user_normalized in question_normalized:
                    score += 35
                elif user_normalized in question_processed:
                    score += 30

            # Reverse substring matching
            if len(question_normalized) > 5:
                if question_normalized in user_normalized:
                    score += 25

            # === PATTERN MATCHING ===
            # Question word patterns
            question_patterns = [
                (['gì', 'nào'], ['gì', 'nào', 'những', 'các']),
                (['bao', 'nhiêu'], ['bao', 'nhiêu', 'mấy']),
                (['khi', 'nào'], ['khi', 'nào', 'lúc', 'thời']),
                (['ở', 'đâu'], ['ở', 'đâu', 'chỗ', 'nơi']),
                (['như', 'thế', 'nào'], ['như', 'thế', 'nào', 'cách', 'làm'])
            ]

            user_set = set(user_words)
            question_set = set(question_words)

            for user_pattern, question_pattern in question_patterns:
                if any(word in user_set for word in user_pattern):
                    if any(word in question_set for word in question_pattern):
                        score += 18

            # === SEMANTIC PHRASE MATCHING ===
            # Key concept matching
            key_concepts = {
                'education': ['học', 'ngành', 'môn', 'giáo', 'dục'],
                'cost': ['phí', 'tiền', 'chi', 'phí', 'giá'],
                'exam': ['thi', 'kiểm', 'tra', 'điểm'],
                'student': ['sinh', 'viên', 'học', 'sinh'],
                'time': ['thời', 'gian', 'giờ', 'phút'],
                'place': ['phòng', 'nơi', 'chỗ', 'địa']
            }

            user_concepts = set()
            question_concepts = set()

            for concept, words in key_concepts.items():
                if any(word in user_set for word in words):
                    user_concepts.add(concept)
                if any(word in question_set for word in words):
                    question_concepts.add(concept)

            concept_overlap = len(user_concepts.intersection(question_concepts))
            score += concept_overlap * 8

            # === STRUCTURE SIMILARITY ===
            # Similar question structure
            if len(user_words) > 0 and len(question_words) > 0:
                # Length similarity
                len_ratio = min(len(user_words), len(question_words)) / max(len(user_words), len(question_words))
                score += len_ratio * 10

                # Position-based matching (important words in similar positions)
                position_score = 0
                for i, word in enumerate(user_words):
                    if word in important_phrases or len(word) > 3:
                        # Check if word appears in similar position in question
                        relative_pos = i / len(user_words)
                        for j, q_word in enumerate(question_words):
                            if word == q_word:
                                q_relative_pos = j / len(question_words)
                                if abs(relative_pos - q_relative_pos) < 0.3:
                                    position_score += 5
                                break

                score += position_score

            # === FINAL SCORING ===
            if score > best_score and score > 15:
                best_score = score
                best_match = row['answer']

        return best_match

    async def fuzzy_match(self, user_input: str) -> Optional[str]:
        """Enhanced fuzzy matching với multiple similarity metrics"""
        await asyncio.sleep(0)  # Yield control to event loop
        user_processed = self.preprocess_text(user_input)
        user_normalized = self.normalize_question(user_input)
        user_words = set(user_processed.split())
        user_norm_words = set(user_normalized.split())

        best_match = None
        best_similarity = 0

        for _, row in self.data.iterrows():
            question_words = set(row['processed_question'].split())
            norm_question_words = set(row['normalized_question'].split())

            if user_words and question_words:
                # Multiple similarity metrics

                # 1. Jaccard similarity
                intersection = len(user_words & question_words)
                union = len(user_words | question_words)
                jaccard = intersection / union if union > 0 else 0

                # 2. Containment similarity (both directions)
                containment1 = intersection / len(user_words) if len(user_words) > 0 else 0
                containment2 = intersection / len(question_words) if len(question_words) > 0 else 0

                # 3. Normalized Jaccard
                norm_intersection = len(user_norm_words & norm_question_words)
                norm_union = len(user_norm_words | norm_question_words)
                norm_jaccard = norm_intersection / norm_union if norm_union > 0 else 0

                # 4. Dice coefficient
                dice = 2 * intersection / (len(user_words) + len(question_words)) if (len(user_words) + len(question_words)) > 0 else 0

                # Combined similarity
                similarity = (
                        jaccard * 0.3 +
                        containment1 * 0.25 +
                        containment2 * 0.15 +
                        norm_jaccard * 0.2 +
                        dice * 0.1
                )

                # Bonus for exact matches of important words
                important_exact = 0
                important_words = ['ngành', 'học', 'phí', 'điểm', 'thi', 'phòng', 'fpt', 'môn']
                for word in important_words:
                    if word in user_words and word in question_words:
                        important_exact += 0.05

                similarity += important_exact

                if similarity > best_similarity and similarity > 0.15:
                    best_similarity = similarity
                    best_match = row['answer']

        return best_match

    def calculate_confidence_score(self, user_input: str, answer: str, method_name: str = '') -> float:
        """Tính confidence score cho câu trả lời dựa trên độ tương đồng với training data"""
        if not answer or not user_input:
            return 0.0

        user_processed = self.preprocess_text(user_input)
        user_normalized = self.normalize_question(user_input)
        user_words = set(user_processed.split())

        # Tìm câu hỏi trong training data có cùng answer
        matching_questions = self.data[self.data['answer'] == answer]

        if len(matching_questions) == 0:
            return 0.0

        max_confidence = 0.0

        for _, row in matching_questions.iterrows():
            question_processed = row['processed_question']
            question_normalized = row['normalized_question']
            question_words = set(question_processed.split())

            # Tính nhiều metric confidence
            confidences = []

            # 1. Jaccard similarity
            if user_words and question_words:
                intersection = len(user_words & question_words)
                union = len(user_words | question_words)
                jaccard = intersection / union if union > 0 else 0
                confidences.append(jaccard)

            # 2. Containment similarity
            if user_words and question_words:
                containment = len(user_words & question_words) / len(user_words) if len(user_words) > 0 else 0
                confidences.append(containment)

            # 3. Exact phrase matching
            if user_normalized and question_normalized:
                # Kiểm tra substring matches
                if user_normalized in question_normalized or question_normalized in user_normalized:
                    confidences.append(0.8)
                else:
                    # N-gram matching
                    user_ngrams = set()
                    question_ngrams = set()

                    user_words_list = user_processed.split()
                    question_words_list = question_processed.split()

                    # Tạo 2-grams và 3-grams
                    for n in range(2, 4):
                        for i in range(len(user_words_list) - n + 1):
                            user_ngrams.add(' '.join(user_words_list[i:i+n]))
                        for i in range(len(question_words_list) - n + 1):
                            question_ngrams.add(' '.join(question_words_list[i:i+n]))

                    if user_ngrams and question_ngrams:
                        ngram_similarity = len(user_ngrams & question_ngrams) / len(user_ngrams | question_ngrams)
                        confidences.append(ngram_similarity)

            # 4. Important word matching
            important_words = {
                'ngành': 1.0, 'học': 1.0, 'phí': 1.0, 'điểm': 1.0, 'thi': 1.0,
                'phòng': 1.0, 'sinh': 1.0, 'viên': 1.0, 'fpt': 1.0, 'thời': 1.0,
                'gian': 1.0, 'bao': 1.0, 'nhiêu': 1.0, 'cách': 1.0, 'quy': 1.0,
                'định': 1.0, 'thực': 1.0, 'tập': 1.0, 'ojt': 1.0
            }

            important_matches = 0
            total_important = 0

            for word in user_words:
                if word in important_words:
                    total_important += 1
                    if word in question_words:
                        important_matches += 1

            if total_important > 0:
                important_confidence = important_matches / total_important
                confidences.append(important_confidence)

            # 5. TF-IDF similarity
            try:
                # Sử dụng TF-IDF matrix 1 (processed + keywords)
                combined_user = f"{user_processed} {self.extract_keywords_from_text(user_input)}"
                combined_question = f"{question_processed} {row['keywords']}"

                user_vector = self.tfidf_vectorizer1.transform([combined_user])
                question_vector = self.tfidf_vectorizer1.transform([combined_question])

                tfidf_similarity = cosine_similarity(user_vector, question_vector)[0][0]
                confidences.append(tfidf_similarity)
            except:
                pass

            # Tính confidence tổng hợp
            if confidences:
                # Weighted average với emphasis trên các metric quan trọng
                weights = [0.25, 0.2, 0.2, 0.2, 0.15]  # Jaccard, Containment, Phrase, Important, TF-IDF

                weighted_confidence = 0.0
                total_weight = 0.0

                for i, conf in enumerate(confidences):
                    if i < len(weights):
                        weighted_confidence += conf * weights[i]
                        total_weight += weights[i]
                    else:
                        weighted_confidence += conf * 0.1
                        total_weight += 0.1

                if total_weight > 0:
                    final_confidence = weighted_confidence / total_weight

                    # Bonus cho exact matches
                    if any(conf > 0.9 for conf in confidences):
                        final_confidence += 0.1

                    # Bonus cho method cụ thể
                    if method_name == 'exact_match':
                        final_confidence += 0.15
                    elif method_name == 'advanced_keyword_match':
                        final_confidence += 0.1
                    elif method_name == 'wildcard_match':
                        final_confidence += 0.08

                    max_confidence = max(max_confidence, final_confidence)

        return min(max_confidence, 1.0)  # Cap at 1.0

    def extract_keywords_from_text(self, text: str) -> str:
        """Trích xuất keywords từ text để tăng accuracy của confidence scoring"""
        processed = self.preprocess_text(text)
        words = processed.split()

        # Lấy important words
        important_words = {
            'ngành', 'học', 'phí', 'điểm', 'thi', 'phòng', 'sinh', 'viên',
            'fpt', 'đại', 'trường', 'môn', 'lịch', 'giảng', 'viên', 'kỳ',
            'học', 'kì', 'cuối', 'giữa', 'midterm', 'final'
        }

        keywords = [word for word in words if word in important_words]
        return ' '.join(keywords)

    def detect_irrelevant_question(self, user_input: str) -> bool:
        """Detect câu hỏi không liên quan đến domain giáo dục - conservative approach"""
        user_processed = self.preprocess_text(user_input)
        user_words = set(user_processed.split())

        # Từ khóa core của domain giáo dục
        education_core_keywords = {
            'ngành', 'học', 'phí', 'điểm', 'thi', 'phòng', 'sinh', 'viên',
            'fpt', 'đại', 'trường', 'môn', 'lịch', 'giảng', 'viên', 'kỳ',
            'kì', 'cuối', 'giữa', 'midterm', 'final', 'thực', 'tập', 'ojt',
            'quy', 'định', 'luật', 'chính', 'sách', 'yêu', 'cầu', 'điều', 'kiện',
            'bao', 'nhiêu', 'cách', 'như', 'thế', 'nào', 'ở', 'đâu', 'khi'
        }

        # Từ khóa rõ ràng không liên quan (chỉ những từ cực kỳ rõ ràng)
        clearly_irrelevant = {
            'thời', 'tiết', 'cà', 'phê', 'nấu', 'ăn', 'phở', 'bún', 'cơm',
            'xe', 'máy', 'ô', 'tô', 'bóng', 'đá', 'ca', 'nhạc', 'phim',
            'game', 'chơi', 'siêu', 'thị', 'mua', 'bán', 'quần', 'áo',
            'du', 'lịch', 'khách', 'sạn', 'máy', 'bay', 'tàu', 'hỏa',
            'yêu', 'đương', 'hẹn', 'hò', 'cưới', 'xin', 'chào'
        }

        # Chỉ reject nếu có từ khóa rõ ràng không liên quan VÀ HOÀN TOÀN không có từ khóa giáo dục
        has_irrelevant = len(user_words.intersection(clearly_irrelevant)) > 0
        has_education = len(user_words.intersection(education_core_keywords)) > 0

        # Chỉ reject những trường hợp cực kỳ rõ ràng
        if has_irrelevant and not has_education and len(user_words) > 2:
            return True

        # Câu hỏi 1 từ hoặc rỗng
        if len(user_words) <= 1:
            return True

        # Câu hỏi chỉ có greeting mà không có nội dung
        greeting_only = {'xin', 'chào', 'hello', 'hi', 'hey'}
        if user_words.issubset(greeting_only):
            return True

        return False

    def validate_answer_quality(self, user_input: str, answer: str, quality_threshold: float = 0.3) -> bool:
        """
        Kiểm tra chất lượng câu trả lời so với câu hỏi
        Returns True nếu câu trả lời phù hợp (relevance >= threshold)
        """
        try:
            # Preprocess both texts
            user_processed = self.preprocess_text(user_input)
            answer_processed = self.preprocess_text(answer)

            # Extract keywords
            user_words = set(user_processed.split())
            answer_words = set(answer_processed.split())

            if not user_words or not answer_words:
                return False

            # 1. Keyword overlap score
            intersection = len(user_words & answer_words)
            union = len(user_words | answer_words)
            keyword_overlap = intersection / union if union > 0 else 0

            # 2. Important word matching
            important_words = {
                'ngành', 'học', 'phí', 'điểm', 'thi', 'phòng', 'sinh', 'viên',
                'fpt', 'thời', 'gian', 'bao', 'nhiêu', 'cách', 'quy', 'định',
                'thực', 'tập', 'ojt', 'môn', 'đại', 'trường', 'khi', 'ở', 'đâu'
            }

            user_important = user_words & important_words
            answer_important = answer_words & important_words

            important_overlap = 0
            if user_important:
                important_overlap = len(user_important & answer_important) / len(user_important)

            # 3. TF-IDF similarity
            tfidf_similarity = 0
            try:
                user_vector = self.tfidf_vectorizer1.transform([user_processed])
                answer_vector = self.tfidf_vectorizer1.transform([answer_processed])
                tfidf_similarity = cosine_similarity(user_vector, answer_vector)[0][0]
            except:
                pass

            # 4. Question type matching
            question_words = {'gì', 'nào', 'bao', 'nhiêu', 'khi', 'ở', 'đâu', 'như', 'thế', 'sao'}
            user_has_question = any(word in user_words for word in question_words)

            # Bonus for question type consistency
            question_bonus = 0.1 if user_has_question else 0

            # 5. Length appropriateness (not too short, not too long)
            length_ratio = min(len(answer), 500) / 500  # Normalize to 0-1
            length_bonus = 0.1 if 20 <= len(answer) <= 800 else 0

            # Combined relevance score
            relevance_score = (
                    keyword_overlap * 0.3 +
                    important_overlap * 0.3 +
                    tfidf_similarity * 0.25 +
                    question_bonus * 0.1 +
                    length_bonus * 0.05
            )

            return relevance_score >= quality_threshold

        except Exception as e:
            print(f"Error in validate_answer_quality: {e}")
            return True  # Default to accept if validation fails

    def validate_answer_quality_relaxed(self, user_input: str, answer: str, quality_threshold: float = 0.05) -> bool:
        """
        Ultra-relaxed validation - chỉ reject những câu trả lời rõ ràng sai lệch
        """
        try:
            # Preprocess
            user_processed = self.preprocess_text(user_input)
            answer_processed = self.preprocess_text(answer)

            user_words = set(user_processed.split())
            answer_words = set(answer_processed.split())

            if not user_words or not answer_words:
                return True  # Accept if can't validate

            # Very basic overlap check - chỉ reject nếu hoàn toàn không liên quan
            intersection = len(user_words & answer_words)
            overlap_ratio = intersection / len(user_words) if len(user_words) > 0 else 0

            # Nếu có ít nhất 1 từ chung thì accept
            if intersection >= 1:
                return True

            # Kiểm tra domain consistency - chỉ reject nếu domain hoàn toàn khác
            domain_words = {
                'ngành', 'học', 'phí', 'điểm', 'thi', 'phòng', 'sinh', 'viên',
                'fpt', 'đại', 'trường', 'môn', 'thực', 'tập', 'ojt', 'quy', 'định'
            }

            user_domain = user_words & domain_words
            answer_domain = answer_words & domain_words

            # Nếu user hỏi về domain education mà answer hoàn toàn không có từ nào liên quan
            if len(user_domain) >= 2 and len(answer_domain) == 0:
                return False

            # Chỉ reject nếu overlap ratio cực kỳ thấp
            if overlap_ratio < quality_threshold:
                return False

            return True

        except Exception as e:
            print(f"Error in validate_answer_quality_relaxed: {e}")
            return True  # Default accept để không giảm accuracy

    async def get_high_quality_response(self, user_input: str) -> str:
        """
        Lấy câu trả lời chất lượng cao với minimal validation để giữ accuracy
        """
        await asyncio.sleep(0)  # Yield control to event loop

        # Kiểm tra domain relevance với conservative approach
        if self.detect_irrelevant_question(user_input):
            return "Xin lỗi, tôi không tìm thấy thông tin phù hợp với câu hỏi của bạn. Vui lòng liên hệ phòng DVSV để được hỗ trợ thêm."

        # Thử từng method theo thứ tự ưu tiên
        methods = [
            ('exact_match', self.exact_match),
            ('hybrid_top_k_search', self.hybrid_top_k_search),
            ('advanced_keyword_match', self.advanced_keyword_match),
            ('wildcard_match', self.wildcard_match),
            ('phrase_match', self.phrase_match),
            ('fuzzy_match', self.fuzzy_match)
        ]

        # Lưu trữ answers để fallback
        candidate_answers = []

        for method_name, method_func in methods:
            try:
                answer = await method_func(user_input)
                if answer:
                    candidate_answers.append((answer, method_name))

                    # Chỉ validate với threshold cực kỳ thấp
                    if self.validate_answer_quality_relaxed(user_input, answer, quality_threshold=0.05):
                        return answer
                    # Nếu không pass validation, vẫn lưu lại làm candidate
            except Exception as e:
                print(f"Error in {method_name}: {e}")
                continue

        # Nếu có candidates nhưng không pass validation, trả về candidate đầu tiên
        # (ưu tiên accuracy hơn precision)
        if candidate_answers:
            return candidate_answers[0][0]

        # Chỉ fallback nếu thật sự không có answer nào
        return "Xin lỗi, tôi không tìm thấy thông tin phù hợp với câu hỏi của bạn. Vui lòng liên hệ phòng DVSV để được hỗ trợ thêm."

    async def get_response(self, user_input: str) -> str:
        """Optimized response với answer quality validation"""
        return await self.get_high_quality_response(user_input)
