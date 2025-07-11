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

class ChatbotAccuracyTester:
    def __init__(self, api_url="http://localhost:8000/chat"):
        self.api_url = api_url
        self.results = []
        
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
        """Tính nhiều loại điểm tương đồng"""
        
        # Preprocess texts
        response_clean = self.preprocess_text(response)
        expected_clean = self.preprocess_text(expected)
        
        if not response_clean or not expected_clean:
            return {
                'sequence_similarity': 0.0,
                'tfidf_similarity': 0.0,
                'word_overlap': 0.0,
                'length_similarity': 0.0
            }
        
        # 1. Sequence similarity (SequenceMatcher)
        sequence_sim = SequenceMatcher(None, response_clean, expected_clean).ratio()
        
        # 2. TF-IDF Cosine similarity
        try:
            vectorizer = TfidfVectorizer().fit([response_clean, expected_clean])
            vectors = vectorizer.transform([response_clean, expected_clean])
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
        len_response = len(response_clean)
        len_expected = len(expected_clean)
        if len_expected > 0:
            length_ratio = min(len_response, len_expected) / max(len_response, len_expected)
        else:
            length_ratio = 0.0
        
        return {
            'sequence_similarity': sequence_sim,
            'tfidf_similarity': tfidf_sim,
            'word_overlap': word_overlap,
            'length_similarity': length_ratio
        }
    
    def calculate_composite_score(self, similarities):
        """Tính điểm tổng hợp từ các điểm similarity"""
        weights = {
            'sequence_similarity': 0.3,
            'tfidf_similarity': 0.4,
            'word_overlap': 0.2,
            'length_similarity': 0.1
        }
        
        composite_score = sum(similarities[key] * weights[key] for key in weights)
        return composite_score
    
    def classify_result(self, composite_score, similarities):
        """Phân loại kết quả dựa trên điểm số"""
        if composite_score >= 0.8:
            return "EXCELLENT"
        elif composite_score >= 0.6:
            return "GOOD"
        elif composite_score >= 0.4:
            return "FAIR"
        elif composite_score >= 0.2:
            return "POOR"
        else:
            return "VERY_POOR"
    
    def call_api(self, question, timeout=30):
        """Gọi API chatbot"""
        try:
            # Kiểm tra question có hợp lệ không
            if not question or not str(question).strip():
                return "Empty question", False, "Question is empty"
            
            payload = {
                "chatInput": str(question).strip(),
                "sessionId": "accuracy_test_session"
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('output', ''), True, None
            else:
                return f"API Error: {response.status_code}", False, f"Status: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "Timeout", False, "Request timeout"
        except requests.exceptions.ConnectionError:
            return "Connection Error", False, "Cannot connect to API"
        except Exception as e:
            return f"Error: {str(e)}", False, str(e)
    
    def test_accuracy(self, testcase_file="testcase.xlsx"):
        """Chạy test accuracy với file testcase"""
        
        print(f"🚀 Bắt đầu test accuracy chatbot")
        print(f"📊 API URL: {self.api_url}")
        print(f"📁 Test file: {testcase_file}")
        
        # Đọc file testcase
        try:
            df = pd.read_excel(testcase_file)
            print(f"✅ Đọc thành công {len(df)} test cases")
            print(f"📋 Columns: {list(df.columns)}")
        except Exception as e:
            print(f"❌ Lỗi đọc file: {e}")
            return
        
        # Kiểm tra columns cần thiết
        required_columns = ['question', 'answer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Thiếu columns: {missing_columns}")
            return
        
        # Kiểm tra API có hoạt động không
        print("\n🔍 Kiểm tra API...")
        test_response, success, error = self.call_api("Hello test", timeout=10)
        if not success:
            print(f"❌ API không hoạt động: {error}")
            return
        print("✅ API hoạt động bình thường")
        
        # Chạy test
        print(f"\n🧪 Bắt đầu test {len(df)} câu hỏi...")
        
        for idx, row in df.iterrows():
            question = str(row['question']) if pd.notna(row['question']) else ""
            expected_answer = str(row['answer']) if pd.notna(row['answer']) else ""
            context = str(row.get('context', '')) if pd.notna(row.get('context', '')) else ""
            
            # Bỏ qua test nếu question hoặc answer rỗng
            if not question.strip() or not expected_answer.strip():
                print(f"\n⚠️  Test {idx + 1}/{len(df)}: Dữ liệu rỗng - bỏ qua")
                continue
            
            print(f"\n📝 Test {idx + 1}/{len(df)}: {question[:50]}...")
            
            # Gọi API
            api_response, success, error = self.call_api(question)
            
            if success:
                # Tính các điểm similarity
                similarities = self.calculate_similarity_scores(api_response, expected_answer)
                composite_score = self.calculate_composite_score(similarities)
                result_class = self.classify_result(composite_score, similarities)
                
                # Lưu kết quả
                result_data = {
                    'test_index': idx + 1,
                    'question': question,
                    'expected_answer': expected_answer,
                    'api_response': api_response,
                    'context': context,
                    'sequence_similarity': similarities['sequence_similarity'],
                    'tfidf_similarity': similarities['tfidf_similarity'],
                    'word_overlap': similarities['word_overlap'],
                    'length_similarity': similarities['length_similarity'],
                    'composite_score': composite_score,
                    'result_class': result_class,
                    'api_success': True,
                    'api_error': None
                }
                
                print(f"   📊 Score: {composite_score:.3f} ({result_class})")
                print(f"   🔍 TF-IDF: {similarities['tfidf_similarity']:.3f}, Word Overlap: {similarities['word_overlap']:.3f}")
                
            else:
                # API lỗi
                result_data = {
                    'test_index': idx + 1,
                    'question': question,
                    'expected_answer': expected_answer,
                    'api_response': api_response,
                    'context': context,
                    'sequence_similarity': 0.0,
                    'tfidf_similarity': 0.0,
                    'word_overlap': 0.0,
                    'length_similarity': 0.0,
                    'composite_score': 0.0,
                    'result_class': 'API_ERROR',
                    'api_success': False,
                    'api_error': error
                }
                
                print(f"   ❌ Lỗi API: {error}")
            
            self.results.append(result_data)
            
            # Nghỉ 1 giây giữa các request
            time.sleep(1)
        
        # Tạo báo cáo
        self.generate_accuracy_report()
    
    def generate_accuracy_report(self):
        """Tạo báo cáo accuracy và xuất file Excel"""
        
        if not self.results:
            print("❌ Không có kết quả để tạo báo cáo")
            return
        
        # Tạo DataFrame từ kết quả
        df_results = pd.DataFrame(self.results)
        
        # Thống kê tổng quát
        total_tests = len(df_results)
        api_errors = len(df_results[df_results['result_class'] == 'API_ERROR'])
        successful_tests = total_tests - api_errors
        
        print(f"\n📊 BÁOCÁO ACCURACY:")
        print(f"   📈 Tổng số test: {total_tests}")
        print(f"   🔥 API errors: {api_errors}")
        print(f"   ✅ Successful tests: {successful_tests}")
        
        if successful_tests > 0:
            # Phân loại kết quả
            excellent = len(df_results[df_results['result_class'] == 'EXCELLENT'])
            good = len(df_results[df_results['result_class'] == 'GOOD'])
            fair = len(df_results[df_results['result_class'] == 'FAIR'])
            poor = len(df_results[df_results['result_class'] == 'POOR'])
            very_poor = len(df_results[df_results['result_class'] == 'VERY_POOR'])
            
            print(f"\n🎯 PHÂN LOẠI KẾT QUẢ:")
            print(f"   🌟 EXCELLENT (≥0.8): {excellent} ({excellent/successful_tests*100:.1f}%)")
            print(f"   ✅ GOOD (≥0.6): {good} ({good/successful_tests*100:.1f}%)")
            print(f"   🔶 FAIR (≥0.4): {fair} ({fair/successful_tests*100:.1f}%)")
            print(f"   ⚠️  POOR (≥0.2): {poor} ({poor/successful_tests*100:.1f}%)")
            print(f"   ❌ VERY_POOR (<0.2): {very_poor} ({very_poor/successful_tests*100:.1f}%)")
            
            # Tính accuracy tổng hợp
            avg_score = df_results[df_results['result_class'] != 'API_ERROR']['composite_score'].mean()
            
            # Accuracy có trọng số: EXCELLENT=1, GOOD=0.8, FAIR=0.6, POOR=0.4, VERY_POOR=0.2
            weighted_score = (excellent * 1.0 + good * 0.8 + fair * 0.6 + poor * 0.4 + very_poor * 0.2) / successful_tests
            
            print(f"\n📊 ĐIỂM SỐ TỔNG HỢP:")
            print(f"   🎯 Average Composite Score: {avg_score:.3f}")
            print(f"   🏆 Weighted Accuracy: {weighted_score:.3f} ({weighted_score*100:.1f}%)")
            
            # Top similarity scores
            print(f"\n📈 ĐIỂM SIMILARITY TRUNG BÌNH:")
            successful_df = df_results[df_results['result_class'] != 'API_ERROR']
            print(f"   🔤 TF-IDF Similarity: {successful_df['tfidf_similarity'].mean():.3f}")
            print(f"   📝 Sequence Similarity: {successful_df['sequence_similarity'].mean():.3f}")
            print(f"   🔗 Word Overlap: {successful_df['word_overlap'].mean():.3f}")
            print(f"   📏 Length Similarity: {successful_df['length_similarity'].mean():.3f}")
        
        # Lưu kết quả chi tiết
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"chatbot_accuracy_test_{timestamp}.xlsx"
        
        # Tạo file Excel với multiple sheets
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Sheet 1: Kết quả chi tiết
            df_results.to_excel(writer, sheet_name='Detailed_Results', index=False)
            
            # Sheet 2: Thống kê tóm tắt
            if successful_tests > 0:
                summary_data = {
                    'Metric': [
                        'Total Tests', 'API Errors', 'Successful Tests',
                        'EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'VERY_POOR',
                        'Average Composite Score', 'Weighted Accuracy',
                        'Avg TF-IDF Similarity', 'Avg Sequence Similarity',
                        'Avg Word Overlap', 'Avg Length Similarity'
                    ],
                    'Value': [
                        total_tests, api_errors, successful_tests,
                        excellent, good, fair, poor, very_poor,
                        f"{avg_score:.3f}", f"{weighted_score:.3f}",
                        f"{successful_df['tfidf_similarity'].mean():.3f}",
                        f"{successful_df['sequence_similarity'].mean():.3f}",
                        f"{successful_df['word_overlap'].mean():.3f}",
                        f"{successful_df['length_similarity'].mean():.3f}"
                    ],
                    'Percentage': [
                        '100%', f"{api_errors/total_tests*100:.1f}%", f"{successful_tests/total_tests*100:.1f}%",
                        f"{excellent/successful_tests*100:.1f}%", f"{good/successful_tests*100:.1f}%",
                        f"{fair/successful_tests*100:.1f}%", f"{poor/successful_tests*100:.1f}%",
                        f"{very_poor/successful_tests*100:.1f}%",
                        f"{avg_score*100:.1f}%", f"{weighted_score*100:.1f}%",
                        f"{successful_df['tfidf_similarity'].mean()*100:.1f}%",
                        f"{successful_df['sequence_similarity'].mean()*100:.1f}%",
                        f"{successful_df['word_overlap'].mean()*100:.1f}%",
                        f"{successful_df['length_similarity'].mean()*100:.1f}%"
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"\n💾 Kết quả đã lưu: {output_file}")
        
        # Lưu báo cáo text
        summary_file = f"chatbot_accuracy_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("CHATBOT ACCURACY TEST REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"API URL: {self.api_url}\n")
            f.write(f"Total tests: {total_tests}\n")
            f.write(f"API errors: {api_errors}\n")
            f.write(f"Successful tests: {successful_tests}\n\n")
            
            if successful_tests > 0:
                f.write("CLASSIFICATION RESULTS:\n")
                f.write(f"EXCELLENT (≥0.8): {excellent} ({excellent/successful_tests*100:.1f}%)\n")
                f.write(f"GOOD (≥0.6): {good} ({good/successful_tests*100:.1f}%)\n")
                f.write(f"FAIR (≥0.4): {fair} ({fair/successful_tests*100:.1f}%)\n")
                f.write(f"POOR (≥0.2): {poor} ({poor/successful_tests*100:.1f}%)\n")
                f.write(f"VERY_POOR (<0.2): {very_poor} ({very_poor/successful_tests*100:.1f}%)\n\n")
                f.write(f"OVERALL SCORES:\n")
                f.write(f"Average Composite Score: {avg_score:.3f}\n")
                f.write(f"Weighted Accuracy: {weighted_score:.3f} ({weighted_score*100:.1f}%)\n")
        
        print(f"📋 Báo cáo tóm tắt: {summary_file}")

if __name__ == "__main__":
    # Tạo tester
    tester = ChatbotAccuracyTester()
    
    # Chạy test
    tester.test_accuracy("testcase.xlsx")
