#!/usr/bin/env python3
"""
Script đo độ chính xác (accuracy) của Chatbot API
So sánh response từ API với expected answers trong testcase
"""

import pandas as pd
import requests
import json
import time
from datetime import datetime
from difflib import SequenceMatcher
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Script này test accuracy bằng cách:
# 1. Gọi API chatbot với câu hỏi từ testcase
# 2. So sánh response với expected answer
# 3. Tính toán nhiều loại similarity scores
# 4. Đánh giá độ chính xác tổng thể

class ChatbotAccuracyTester:
    def __init__(self, api_url="http://localhost:8000/chat"):
        self.api_url = api_url
        self.results = []
        
        # Initialize sentence transformer cho semantic similarity
        print("🔄 Loading SentenceTransformer model...")
        try:
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("✅ SentenceTransformer loaded successfully")
        except:
            print("⚠️  SentenceTransformer failed to load, using TF-IDF only")
            self.sentence_model = None
        
    def preprocess_text(self, text):
        """Tiền xử lý văn bản để so sánh"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower().strip()
        # Loại bỏ dấu câu và ký tự đặc biệt
        text = re.sub(r'[^\w\s]', ' ', text)
        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def calculate_similarity_scores(self, response, expected):
        """Tính toán nhiều loại điểm tương đồng giữa response và expected answer"""
        
        # Preprocess texts
        response_clean = self.preprocess_text(response)
        expected_clean = self.preprocess_text(expected)
        
        if not response_clean or not expected_clean:
            return {
                'sequence_similarity': 0.0,
                'tfidf_similarity': 0.0,
                'word_overlap': 0.0,
                'length_similarity': 0.0,
                'semantic_similarity': 0.0,
                'keyword_match': 0.0,
                'exact_match': 0.0
            }
        
        # 1. Sequence similarity (SequenceMatcher)
        sequence_sim = SequenceMatcher(None, response_clean, expected_clean).ratio()
        
        # 2. TF-IDF Cosine similarity
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
            vectors = vectorizer.fit_transform([response_clean, expected_clean])
            tfidf_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        except:
            tfidf_sim = 0.0
        
        # 3. Word overlap percentage
        response_words = set(response_clean.split())
        expected_words = set(expected_clean.split())
        if len(expected_words) > 0:
            word_overlap = len(response_words.intersection(expected_words)) / len(expected_words)
        else:
            word_overlap = 0.0
        
        # 4. Length similarity (phạt quá ngắn hoặc quá dài)
        len_response = len(response_clean.split())
        len_expected = len(expected_clean.split())
        if len_expected > 0:
            length_ratio = min(len_response, len_expected) / max(len_response, len_expected)
        else:
            length_ratio = 0.0
        
        # 5. Semantic similarity using SentenceTransformer
        semantic_sim = 0.0
        if self.sentence_model:
            try:
                embeddings = self.sentence_model.encode([response, expected])
                semantic_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            except:
                semantic_sim = 0.0
        
        # 6. Keyword matching (quan trọng nhất cho domain-specific answers)
        keyword_sim = self.calculate_keyword_similarity(response_clean, expected_clean)
        
        # 7. Exact match (case insensitive)
        exact_match = 1.0 if response_clean == expected_clean else 0.0
        
        return {
            'sequence_similarity': float(sequence_sim),
            'tfidf_similarity': float(tfidf_sim),
            'word_overlap': float(word_overlap),
            'length_similarity': float(length_ratio),
            'semantic_similarity': float(semantic_sim),
            'keyword_match': float(keyword_sim),
            'exact_match': float(exact_match)
        }
    
    def calculate_keyword_similarity(self, response, expected):
        """Tính similarity dựa trên keywords quan trọng"""
        
        # Định nghĩa keywords quan trọng cho domain giáo dục
        important_keywords = [
            'học phí', 'ngành', 'thi', 'điểm', 'đăng ký', 'thư viện', 'môn học',
            'tín chỉ', 'gpa', 'tốt nghiệp', 'ký túc xá', 'thời khóa biểu',
            'miễn giảm', 'học bổng', 'chuyên ngành', 'khoa', 'lịch thi'
        ]
        
        # Tìm keywords trong cả response và expected
        response_keywords = set()
        expected_keywords = set()
        
        for keyword in important_keywords:
            if keyword in response:
                response_keywords.add(keyword)
            if keyword in expected:
                expected_keywords.add(keyword)
        
        # Tính similarity dựa trên keywords match
        if len(expected_keywords) > 0:
            keyword_overlap = len(response_keywords.intersection(expected_keywords)) / len(expected_keywords)
        else:
            # Nếu không có keyword quan trọng, dùng word overlap thông thường
            response_words = set(response.split())
            expected_words = set(expected.split())
            if len(expected_words) > 0:
                keyword_overlap = len(response_words.intersection(expected_words)) / len(expected_words)
            else:
                keyword_overlap = 0.0
        
        return keyword_overlap
    
    def calculate_composite_score(self, similarities):
        """Tính điểm tổng hợp từ các điểm similarity với trọng số tối ưu"""
        
        # Trọng số được điều chỉnh cho domain giáo dục
        weights = {
            'semantic_similarity': 0.25,    # Ý nghĩa semantic quan trọng nhất
            'keyword_match': 0.25,          # Keywords domain-specific quan trọng
            'tfidf_similarity': 0.20,       # TF-IDF cho content matching
            'word_overlap': 0.15,           # Word overlap cơ bản
            'sequence_similarity': 0.10,    # Sequence similarity
            'length_similarity': 0.05,      # Length penalty
            'exact_match': 0.0              # Bonus cho exact match (rare)
        }
        
        # Thêm bonus cho exact match
        if similarities['exact_match'] > 0:
            return 1.0
        
        # Tính composite score
        composite_score = sum(similarities[key] * weights[key] for key in weights if key in similarities)
        
        # Normalize to ensure score is between 0 and 1
        composite_score = max(0.0, min(1.0, composite_score))
        
        return composite_score
    
    def classify_accuracy(self, composite_score, similarities):
        """Phân loại độ chính xác dựa trên điểm số"""
        
        # Thresholds được điều chỉnh cho chatbot domain-specific
        if composite_score >= 0.85:
            return "EXCELLENT", "🌟"
        elif composite_score >= 0.70:
            return "GOOD", "✅"
        elif composite_score >= 0.55:
            return "FAIR", "🔶"
        elif composite_score >= 0.35:
            return "POOR", "⚠️"
        else:
            return "VERY_POOR", "❌"
    
    def call_api(self, question, timeout=30):
        """Gọi API chatbot và lấy response với improved error handling"""
        try:
            # Validate và clean input
            if not question or not str(question).strip():
                return "Empty question", False, "Question is empty", None
            
            # Clean question text
            cleaned_question = str(question).strip()
            
            # Remove any problematic characters that might cause 400 errors
            cleaned_question = cleaned_question.replace('\x00', '')  # Remove null bytes
            cleaned_question = cleaned_question.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
            cleaned_question = ' '.join(cleaned_question.split())  # Normalize whitespace
            
            # Ensure question is not too long (max 1000 chars to avoid 400 errors)
            if len(cleaned_question) > 1000:
                cleaned_question = cleaned_question[:1000] + "..."
            
            payload = {
                "chatInput": cleaned_question,
                "sessionId": f"accuracy_test_{int(time.time())}"
            }
            
            # Enhanced headers
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "User-Agent": "ChatbotAccuracyTester/1.0"
            }
            
            start_time = time.time()
            
            # Log request for debugging
            print(f"🔍 API Request: {cleaned_question[:50]}...")
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=timeout,
                headers=headers
            )
            response_time = time.time() - start_time
            
            # Log response status
            print(f"📡 API Response: {response.status_code} ({response_time:.2f}s)")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    # Extract main response
                    api_response = result.get('output', '')
                    
                    # Extract additional info if available
                    additional_info = {
                        'response_time': response_time,
                        'route_used': result.get('route', 'unknown'),
                        'similarity_score': result.get('similarity_score', 0.0),
                        'source': result.get('source', ''),
                        'category': result.get('category', 'unknown'),
                        'status_code': response.status_code
                    }
                    
                    return api_response, True, None, additional_info
                except json.JSONDecodeError as e:
                    return f"JSON Parse Error: {str(e)}", False, f"Invalid JSON response", {
                        'response_time': response_time,
                        'status_code': response.status_code
                    }
            
            else:
                # Handle different error codes specifically
                error_msg = f"HTTP {response.status_code}"
                
                # Try to get error details from response
                try:
                    error_details = response.json()
                    if 'detail' in error_details:
                        error_msg += f": {error_details['detail']}"
                    elif 'message' in error_details:
                        error_msg += f": {error_details['message']}"
                except:
                    # If response is not JSON, get text
                    try:
                        error_text = response.text[:200]  # First 200 chars
                        if error_text:
                            error_msg += f": {error_text}"
                    except:
                        pass
                
                print(f"❌ API Error {response.status_code}: {error_msg}")
                print(f"🔍 Request payload: {payload}")
                
                return f"API Error: {error_msg}", False, error_msg, {
                    'response_time': response_time,
                    'status_code': response.status_code,
                    'request_payload': payload
                }
                
        except requests.exceptions.Timeout:
            print(f"⏰ Request timeout after {timeout}s")
            return "Timeout", False, "Request timeout", {'response_time': timeout}
        except requests.exceptions.ConnectionError:
            print(f"🔌 Connection error to {self.api_url}")
            return "Connection Error", False, "Cannot connect to API", {'response_time': 0}
        except Exception as e:
            print(f"💥 Unexpected error: {str(e)}")
            return f"Error: {str(e)}", False, str(e), {'response_time': 0}
    
    def test_accuracy(self, testcase_file="testcase.xlsx", num_samples=None):
        """Chạy test accuracy với file testcase"""
        
        print(f"🚀 CHATBOT ACCURACY TESTING")
        print(f"=" * 50)
        print(f"📊 API URL: {self.api_url}")
        print(f"📁 Test file: {testcase_file}")
        
        # Đọc file testcase
        try:
            df = pd.read_excel(testcase_file)
            print(f"✅ Loaded {len(df)} test cases from {testcase_file}")
            print(f"📋 Columns: {list(df.columns)}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return
        
        # Kiểm tra columns cần thiết
        required_columns = ['question', 'answer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            print(f"📋 Available columns: {list(df.columns)}")
            return
        
        # Lọc dữ liệu hợp lệ
        df_valid = df.dropna(subset=['question', 'answer'])
        df_valid = df_valid[
            (df_valid['question'].str.strip() != '') & 
            (df_valid['answer'].str.strip() != '')
        ]
        
        print(f"📊 Valid test cases: {len(df_valid)}/{len(df)}")
        
        # Lấy sample nếu được chỉ định
        if num_samples and num_samples < len(df_valid):
            df_valid = df_valid.sample(n=num_samples, random_state=42)
            print(f"🎲 Sampling {num_samples} test cases for testing")
        #
        # # Test API connection
        # print(f"\n🔍 Testing API connection...")
        # test_response, success, error, _ = self.call_api("Hello test", timeout=10)
        # if not success:
        #     print(f"❌ API connection failed: {error}")
        #     return
        # print("✅ API connection successful")
        
        # Chạy accuracy test
        print(f"\n🧪 Running accuracy test on {len(df_valid)} questions...")
        print(f"=" * 50)
        
        for idx, row in df_valid.iterrows():
            question = str(row['question']).strip()
            expected_answer = str(row['answer']).strip()
            category = str(row.get('category', 'Unknown')).strip()
            source = str(row.get('nguồn', '')).strip()
            
            print(f"\n📝 Test {len(self.results) + 1}/{len(df_valid)}")
            print(f"❓ Q: {question[:80]}{'...' if len(question) > 80 else ''}")
            print(f"🏷️  Category: {category}")
            
            # Gọi API
            api_response, success, error, additional_info = self.call_api(question)
            
            if success:
                # Tính similarity scores
                similarities = self.calculate_similarity_scores(api_response, expected_answer)
                composite_score = self.calculate_composite_score(similarities)
                accuracy_class, icon = self.classify_accuracy(composite_score, similarities)
                
                # Display results
                print(f"✅ API Response: {api_response[:100]}{'...' if len(api_response) > 100 else ''}")
                print(f"📊 Accuracy: {composite_score:.3f} {icon} {accuracy_class}")
                print(f"🔍 Semantic: {similarities['semantic_similarity']:.3f}, Keywords: {similarities['keyword_match']:.3f}, TF-IDF: {similarities['tfidf_similarity']:.3f}")
                print(f"⏱️  Response time: {additional_info.get('response_time', 0):.2f}s")
                if additional_info.get('route_used'):
                    print(f"🛤️  Route: {additional_info['route_used']}")
                
                # Lưu kết quả chi tiết
                result_data = {
                    'test_index': len(self.results) + 1,
                    'question': question,
                    'expected_answer': expected_answer,
                    'api_response': api_response,
                    'category': category,
                    'source': source,
                    'sequence_similarity': similarities['sequence_similarity'],
                    'tfidf_similarity': similarities['tfidf_similarity'],
                    'word_overlap': similarities['word_overlap'],
                    'length_similarity': similarities['length_similarity'],
                    'semantic_similarity': similarities['semantic_similarity'],
                    'keyword_match': similarities['keyword_match'],
                    'exact_match': similarities['exact_match'],
                    'composite_score': composite_score,
                    'accuracy_class': accuracy_class,
                    'api_success': True,
                    'api_error': None,
                    'response_time': additional_info.get('response_time', 0),
                    'route_used': additional_info.get('route_used', 'unknown'),
                    'api_similarity_score': additional_info.get('similarity_score', 0.0)
                }
                
            else:
                # API error
                print(f"❌ API Error: {error}")
                
                result_data = {
                    'test_index': len(self.results) + 1,
                    'question': question,
                    'expected_answer': expected_answer,
                    'api_response': api_response,
                    'category': category,
                    'source': source,
                    'sequence_similarity': 0.0,
                    'tfidf_similarity': 0.0,
                    'word_overlap': 0.0,
                    'length_similarity': 0.0,
                    'semantic_similarity': 0.0,
                    'keyword_match': 0.0,
                    'exact_match': 0.0,
                    'composite_score': 0.0,
                    'accuracy_class': 'API_ERROR',
                    'api_success': False,
                    'api_error': error,
                    'response_time': additional_info.get('response_time', 0),
                    'route_used': 'error',
                    'api_similarity_score': 0.0
                }
            
            self.results.append(result_data)
            
            # Short delay between requests
            time.sleep(0.5)
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        print(f"\n🎉 TESTING COMPLETED!")
        print(f"📊 Total tests run: {len(self.results)}")
        print(f"💾 Detailed results saved to Excel file")
    
    def generate_comprehensive_report(self):
        """Tạo báo cáo toàn diện về accuracy"""
        
        if not self.results:
            print("❌ No results to generate report")
            return
        
        # Create DataFrame from results
        df_results = pd.DataFrame(self.results)
        
        # Calculate statistics
        total_tests = len(df_results)
        api_errors = len(df_results[df_results['accuracy_class'] == 'API_ERROR'])
        successful_tests = total_tests - api_errors
        
        print(f"\n" + "=" * 60)
        print(f"📊 COMPREHENSIVE ACCURACY REPORT")
        print(f"=" * 60)
        print(f"📈 Total tests: {total_tests}")
        print(f"❌ API errors: {api_errors}")
        print(f"✅ Successful tests: {successful_tests}")
        
        if successful_tests > 0:
            successful_df = df_results[df_results['accuracy_class'] != 'API_ERROR']
            
            # Accuracy classification
            excellent = len(successful_df[successful_df['accuracy_class'] == 'EXCELLENT'])
            good = len(successful_df[successful_df['accuracy_class'] == 'GOOD'])
            fair = len(successful_df[successful_df['accuracy_class'] == 'FAIR'])
            poor = len(successful_df[successful_df['accuracy_class'] == 'POOR'])
            very_poor = len(successful_df[successful_df['accuracy_class'] == 'VERY_POOR'])
            
            print(f"\n🎯 ACCURACY CLASSIFICATION:")
            print(f"   🌟 EXCELLENT (≥0.85): {excellent} ({excellent/successful_tests*100:.1f}%)")
            print(f"   ✅ GOOD (≥0.70): {good} ({good/successful_tests*100:.1f}%)")
            print(f"   🔶 FAIR (≥0.55): {fair} ({fair/successful_tests*100:.1f}%)")
            print(f"   ⚠️  POOR (≥0.35): {poor} ({poor/successful_tests*100:.1f}%)")
            print(f"   ❌ VERY_POOR (<0.35): {very_poor} ({very_poor/successful_tests*100:.1f}%)")
            
            # Overall scores
            avg_composite = successful_df['composite_score'].mean()
            avg_semantic = successful_df['semantic_similarity'].mean()
            avg_keyword = successful_df['keyword_match'].mean()
            avg_tfidf = successful_df['tfidf_similarity'].mean()
            avg_response_time = successful_df['response_time'].mean()
            
            # Weighted accuracy (higher weight for better classes)
            weighted_accuracy = (
                excellent * 1.0 + good * 0.8 + fair * 0.6 + poor * 0.4 + very_poor * 0.2
            ) / successful_tests
            
            print(f"\n📊 OVERALL METRICS:")
            print(f"   🎯 Average Composite Score: {avg_composite:.3f} ({avg_composite*100:.1f}%)")
            print(f"   🏆 Weighted Accuracy: {weighted_accuracy:.3f} ({weighted_accuracy*100:.1f}%)")
            print(f"   🧠 Semantic Similarity: {avg_semantic:.3f} ({avg_semantic*100:.1f}%)")
            print(f"   🔑 Keyword Match: {avg_keyword:.3f} ({avg_keyword*100:.1f}%)")
            print(f"   📝 TF-IDF Similarity: {avg_tfidf:.3f} ({avg_tfidf*100:.1f}%)")
            print(f"   ⏱️  Avg Response Time: {avg_response_time:.2f}s")
            
            # Category-wise analysis if available
            if 'category' in df_results.columns:
                print(f"\n📋 ACCURACY BY CATEGORY:")
                for category in successful_df['category'].unique():
                    if category and category != 'Unknown':
                        cat_df = successful_df[successful_df['category'] == category]
                        cat_avg = cat_df['composite_score'].mean()
                        cat_count = len(cat_df)
                        print(f"   {category}: {cat_avg:.3f} ({cat_count} tests)")
            
            # Route analysis if available
            if 'route_used' in df_results.columns:
                print(f"\n🛤️  ACCURACY BY ROUTE:")
                for route in successful_df['route_used'].unique():
                    if route and route != 'unknown':
                        route_df = successful_df[successful_df['route_used'] == route]
                        route_avg = route_df['composite_score'].mean()
                        route_count = len(route_df)
                        print(f"   {route}: {route_avg:.3f} ({route_count} tests)")
        
        # Save comprehensive results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Excel file with multiple sheets
        excel_file = f"chatbot_accuracy_detailed_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Detailed results
            df_results.to_excel(writer, sheet_name='Detailed_Results', index=False)
            
            # Summary statistics
            if successful_tests > 0:
                summary_data = {
                    'Metric': [
                        'Total Tests', 'API Errors', 'Successful Tests',
                        'EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'VERY_POOR',
                        'Average Composite Score', 'Weighted Accuracy',
                        'Semantic Similarity', 'Keyword Match', 'TF-IDF Similarity',
                        'Average Response Time'
                    ],
                    'Value': [
                        total_tests, api_errors, successful_tests,
                        excellent, good, fair, poor, very_poor,
                        f"{avg_composite:.3f}", f"{weighted_accuracy:.3f}",
                        f"{avg_semantic:.3f}", f"{avg_keyword:.3f}",
                        f"{avg_tfidf:.3f}", f"{avg_response_time:.2f}s"
                    ],
                    'Percentage': [
                        '100%', f"{api_errors/total_tests*100:.1f}%", f"{successful_tests/total_tests*100:.1f}%",
                        f"{excellent/successful_tests*100:.1f}%", f"{good/successful_tests*100:.1f}%",
                        f"{fair/successful_tests*100:.1f}%", f"{poor/successful_tests*100:.1f}%",
                        f"{very_poor/successful_tests*100:.1f}%",
                        f"{avg_composite*100:.1f}%", f"{weighted_accuracy*100:.1f}%",
                        f"{avg_semantic*100:.1f}%", f"{avg_keyword*100:.1f}%",
                        f"{avg_tfidf*100:.1f}%", ""
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
                
                # Category breakdown if available
                if 'category' in df_results.columns:
                    category_stats = []
                    for category in successful_df['category'].unique():
                        if category and category != 'Unknown':
                            cat_df = successful_df[successful_df['category'] == category]
                            category_stats.append({
                                'Category': category,
                                'Test_Count': len(cat_df),
                                'Avg_Composite_Score': cat_df['composite_score'].mean(),
                                'Avg_Semantic': cat_df['semantic_similarity'].mean(),
                                'Avg_Keyword': cat_df['keyword_match'].mean(),
                                'Excellence_Rate': len(cat_df[cat_df['accuracy_class'] == 'EXCELLENT']) / len(cat_df)
                            })
                    
                    if category_stats:
                        category_df = pd.DataFrame(category_stats)
                        category_df.to_excel(writer, sheet_name='Category_Analysis', index=False)
        
        print(f"\n💾 Detailed results saved to: {excel_file}")
        
        # Text summary
        summary_file = f"chatbot_accuracy_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("CHATBOT ACCURACY TEST SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"API URL: {self.api_url}\n")
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"Successful Tests: {successful_tests}\n")
            f.write(f"API Errors: {api_errors}\n\n")
            
            if successful_tests > 0:
                f.write("ACCURACY BREAKDOWN:\n")
                f.write(f"EXCELLENT: {excellent} ({excellent/successful_tests*100:.1f}%)\n")
                f.write(f"GOOD: {good} ({good/successful_tests*100:.1f}%)\n")
                f.write(f"FAIR: {fair} ({fair/successful_tests*100:.1f}%)\n")
                f.write(f"POOR: {poor} ({poor/successful_tests*100:.1f}%)\n")
                f.write(f"VERY_POOR: {very_poor} ({very_poor/successful_tests*100:.1f}%)\n\n")
                f.write(f"OVERALL SCORES:\n")
                f.write(f"Average Composite Score: {avg_composite:.3f}\n")
                f.write(f"Weighted Accuracy: {weighted_accuracy:.3f}\n")
                f.write(f"Semantic Similarity: {avg_semantic:.3f}\n")
                f.write(f"Keyword Match: {avg_keyword:.3f}\n")
        
        print(f"📋 Summary saved to: {summary_file}")
        
        # Performance assessment
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        if successful_tests > 0:
            if weighted_accuracy >= 0.85:
                print("   🌟 EXCELLENT - Chatbot performs exceptionally well!")
            elif weighted_accuracy >= 0.70:
                print("   ✅ GOOD - Chatbot performs well with room for improvement")
            elif weighted_accuracy >= 0.55:
                print("   🔶 FAIR - Chatbot needs significant improvement")
            else:
                print("   ❌ POOR - Chatbot requires major improvements")
            
            print(f"\n💡 RECOMMENDATIONS:")
            if avg_semantic < 0.6:
                print("   • Improve semantic understanding and context processing")
            if avg_keyword < 0.7:
                print("   • Enhance domain-specific keyword recognition")
            if avg_response_time > 3.0:
                print("   • Optimize response time for better user experience")
            if excellent/successful_tests < 0.5:
                print("   • Focus on improving answer quality and relevance")
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'weighted_accuracy': weighted_accuracy if successful_tests > 0 else 0,
            'avg_composite_score': avg_composite if successful_tests > 0 else 0,
            'excel_file': excel_file,
            'summary_file': summary_file
        }

if __name__ == "__main__":
    print("🚀 CHATBOT ACCURACY TESTING SUITE")
    print("=" * 50)
    
    # Initialize tester
    tester = ChatbotAccuracyTester(api_url="http://localhost:8000/chat")
    
    # Test với file testcase.xlsx (test cases)
    print("\n📁 Testing with testcase.xlsx...")
    
    # Option 1: Test toàn bộ testcase
    tester.test_accuracy("testcase.xlsx")
    
    # Option 2: Test với sample nhỏ (uncomment để test nhanh)
    # tester.test_accuracy("testcase.xlsx", num_samples=10)
    
    print("\n✅ Testing completed! Check the generated Excel and text files for detailed results.")
