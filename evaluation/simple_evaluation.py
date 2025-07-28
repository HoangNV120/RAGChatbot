"""
Simple Evaluation Script - Evaluate Master Chatbot qua API
"""

import pandas as pd
import asyncio
import aiohttp
import json
import os
from datetime import datetime
from typing import Dict, List

class SimpleEvaluator:
    def __init__(self):
        """Initialize evaluator với API client"""
        print("🚀 Khởi tạo Simple Evaluator...")
        self.api_url = "http://localhost:8000/chat"
        self.session = None
        self.results = []

    async def initialize_session(self):
        """Khởi tạo HTTP session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            print("✅ HTTP session đã được khởi tạo!")

    async def close_session(self):
        """Đóng HTTP session"""
        if self.session:
            await self.session.close()
            print("✅ HTTP session đã được đóng!")

    async def get_chatbot_response(self, question: str) -> str:
        """Lấy response từ API chatbot"""
        try:
            await self.initialize_session()

            # Tạo payload JSON
            payload = {
                "chatInput": question,
                "sessionId": ""
            }

            # Gọi API
            async with self.session.post(
                self.api_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    response_data = await response.json()
                    # Lấy output từ response JSON (đã sửa từ 'answer' thành 'output')
                    output = response_data.get('output', 'Không có câu trả lời')
                    return output
                else:
                    error_text = await response.text()
                    return f"API Error {response.status}: {error_text}"

        except Exception as e:
            print(f"❌ Lỗi khi gọi API cho câu hỏi: {question[:50]}... - {e}")
            return f"Lỗi: {str(e)}"

    def load_evaluation_data(self, file_path: str) -> pd.DataFrame:
        """Load data evaluation từ Excel file"""
        print(f"📂 Đang load data từ: {file_path}")
        try:
            df = pd.read_excel(file_path)
            print(f"✅ Đã load {len(df)} câu hỏi từ file evaluation")

            # Kiểm tra columns cần thiết
            required_columns = ['question', 'answer']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"⚠️ Thiếu columns: {missing_columns}")
                print(f"Columns có sẵn: {list(df.columns)}")

                # Thử map với các tên column khác
                column_mapping = {
                    'Question': 'question',
                    'câu hỏi': 'question',
                    'cau_hoi': 'question',
                    'Answer': 'answer',
                    'câu trả lời': 'answer',
                    'cau_tra_loi': 'answer',
                    'ground_truth': 'answer',
                    'expected_answer': 'answer'
                }

                for old_name, new_name in column_mapping.items():
                    if old_name in df.columns:
                        df[new_name] = df[old_name]
                        print(f"✅ Đã map column '{old_name}' -> '{new_name}'")

            # Loại bỏ rows có giá trị null
            df = df.dropna(subset=['question', 'answer'])
            print(f"✅ Sau khi loại bỏ null values: {len(df)} câu hỏi")

            return df

        except Exception as e:
            print(f"❌ Lỗi khi load data: {e}")
            raise

    async def evaluate_single_question(self, question: str, ground_truth: str, index: int, total: int) -> Dict:
        """Evaluate một câu hỏi"""
        print(f"🔍 Đang evaluate câu hỏi {index + 1}/{total}: {question[:100]}...")

        try:
            # Lấy response từ API
            response = await self.get_chatbot_response(question)

            # Tạo result record
            result = {
                'question': question,
                'ground_truth': ground_truth,
                'chatbot_response': response,
                'index': index + 1,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            print(f"✅ Hoàn thành câu hỏi {index + 1}/{total}")
            return result

        except Exception as e:
            print(f"❌ Lỗi khi evaluate câu hỏi {index + 1}: {e}")
            return {
                'question': question,
                'ground_truth': ground_truth,
                'chatbot_response': f"Lỗi: {str(e)}",
                'index': index + 1,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    async def run_evaluation(self, input_file: str, output_file: str):
        """Chạy evaluation trên toàn bộ dataset"""
        print("🚀 Bắt đầu Simple Evaluation...")

        try:
            # Load data
            df = self.load_evaluation_data(input_file)

            # Initialize HTTP session
            await self.initialize_session()

            # Test API connection
            print("🔄 Kiểm tra kết nối API...")
            test_response = await self.get_chatbot_response("test")
            if test_response.startswith("Lỗi:") or test_response.startswith("API Error"):
                print("❌ Không thể kết nối tới API. Vui lòng kiểm tra server đang chạy tại localhost:8000")
                return None
            print("✅ Kết nối API thành công!")

            # Evaluate từng câu hỏi
            print(f"🔄 Đang evaluate {len(df)} câu hỏi...")

            results = []
            for index, row in df.iterrows():
                question = str(row['question'])
                ground_truth = str(row['answer'])

                # Evaluate single question
                result = await self.evaluate_single_question(
                    question, ground_truth, index, len(df)
                )
                results.append(result)

                # Progress update
                if (index + 1) % 5 == 0:
                    print(f"📊 Đã hoàn thành {index + 1}/{len(df)} câu hỏi")

                # Small delay to avoid overwhelming the API
                await asyncio.sleep(0.1)

            # Tạo DataFrame từ results
            results_df = pd.DataFrame(results)

            # Reorder columns
            columns_order = ['index', 'question', 'ground_truth', 'chatbot_response', 'timestamp']
            results_df = results_df[columns_order]

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Save to Excel
            print(f"💾 Đang lưu kết quả vào: {output_file}")
            results_df.to_excel(output_file, index=False, engine='openpyxl')

            print(f"✅ Đã lưu kết quả evaluation vào: {output_file}")
            print(f"📊 Tổng số câu hỏi đã evaluate: {len(results_df)}")

            # In một số thống kê cơ bản
            successful_responses = len([r for r in results if not r['chatbot_response'].startswith('Lỗi:') and not r['chatbot_response'].startswith('API Error')])
            error_responses = len(results) - successful_responses

            print(f"📈 Thống kê:")
            print(f"   - Tổng số câu hỏi: {len(results)}")
            print(f"   - Trả lời thành công: {successful_responses}")
            print(f"   - Lỗi: {error_responses}")
            print(f"   - Tỷ lệ thành công: {successful_responses/len(results)*100:.1f}%")

            return results_df

        except Exception as e:
            print(f"❌ Lỗi trong quá trình evaluation: {e}")
            raise
        finally:
            # Đóng session
            await self.close_session()

    def print_sample_results(self, results_df: pd.DataFrame, num_samples: int = 3):
        """In một số kết quả mẫu"""
        print(f"\n📋 Mẫu kết quả evaluation (hiển thị {num_samples} câu đầu):")
        print("=" * 100)

        for i in range(min(num_samples, len(results_df))):
            row = results_df.iloc[i]
            print(f"\n📝 Câu hỏi {i+1}:")
            print(f"   Question: {row['question']}")
            print(f"   Ground Truth: {row['ground_truth'][:200]}...")
            print(f"   Chatbot Response: {row['chatbot_response'][:200]}...")
            print(f"   Timestamp: {row['timestamp']}")
            print("-" * 50)

async def main():
    """Main function để chạy evaluation"""
    print("🎯 Simple Evaluation - Master Chatbot API")
    print("=" * 50)

    # Đường dẫn files - sửa lại đường dẫn
    input_file = "app/data_evaluation.xlsx"
    output_file = f"evaluation/simple_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    # Kiểm tra file input tồn tại
    if not os.path.exists(input_file):
        print(f"❌ File input không tồn tại: {input_file}")

        # Thử các đường dẫn khác
        alternative_paths = [
            "data_evaluation.xlsx",
            "evaluation/data_evaluation.xlsx",
            "app/data_evaluation.xlsx",
            "../app/data_evaluation.xlsx",
            "./app/data_evaluation.xlsx"
        ]

        print("🔍 Đang tìm file trong các đường dẫn khác...")
        for alt_path in alternative_paths:
            print(f"   Kiểm tra: {alt_path}")
            if os.path.exists(alt_path):
                input_file = alt_path
                print(f"✅ Tìm thấy file tại: {alt_path}")
                break
        else:
            print("❌ Không tìm thấy file data_evaluation.xlsx")
            # List files in current directory and app/data
            print("📁 Files trong thư mục hiện tại:")
            try:
                for file in os.listdir('.'):
                    if file.endswith('.xlsx'):
                        print(f"   {file}")
            except:
                pass

            print("📁 Files trong app/data/:")
            try:
                if os.path.exists('app/data'):
                    for file in os.listdir('app/data'):
                        if file.endswith('.xlsx'):
                            print(f"   app/data/{file}")
            except:
                pass
            return

    try:
        # Tạo evaluator
        evaluator = SimpleEvaluator()

        # Chạy evaluation
        results_df = await evaluator.run_evaluation(input_file, output_file)

        if results_df is not None:
            # In sample results
            evaluator.print_sample_results(results_df)
            print(f"\n✅ Evaluation hoàn thành! Kết quả đã được lưu vào: {output_file}")
        else:
            print("❌ Evaluation không thành công do lỗi API")

    except Exception as e:
        print(f"❌ Lỗi trong quá trình evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Chạy evaluation
    asyncio.run(main())
