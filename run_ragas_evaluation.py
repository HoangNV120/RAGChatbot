"""
Script chạy đơn giản cho RAGAS Evaluation
Chạy: python run_ragas_evaluation.py
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation_ragas import RAGASEvaluator

async def run_evaluation():
    """
    Chạy đánh giá RAGAS với các tùy chọn
    """
    print("🚀 RAGAS Evaluation for RAG Chatbot System")
    print("=" * 50)

    # Kiểm tra file testcase.xlsx có tồn tại không
    testcase_file = "testcase.xlsx"
    if not os.path.exists(testcase_file):
        print(f"❌ File {testcase_file} không tồn tại!")
        print("Vui lòng đảm bảo file testcase.xlsx có các cột: question, answer")
        return

    # Khởi tạo evaluator
    evaluator = RAGASEvaluator()

    try:
        print(f"📊 Đang đọc file {testcase_file}...")

        # Chạy evaluation
        results = await evaluator.run_evaluation(testcase_file)

        print(f"\n✅ Đã hoàn thành đánh giá {len(results)} câu hỏi!")

        # Export kết quả
        evaluator.export_results()

        print("\n🎉 Evaluation hoàn thành thành công!")
        print("📁 Kết quả đã được xuất ra file Excel")

    except FileNotFoundError as e:
        print(f"❌ Không tìm thấy file: {e}")
        print("Vui lòng kiểm tra đường dẫn file testcase.xlsx")

    except KeyError as e:
        print(f"❌ Thiếu cột trong file Excel: {e}")
        print("File testcase.xlsx cần có các cột: question, answer")

    except Exception as e:
        print(f"❌ Lỗi khi chạy evaluation: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function
    """
    try:
        # Chạy async evaluation
        asyncio.run(run_evaluation())

    except KeyboardInterrupt:
        print("\n⚠️ Evaluation bị dừng bởi người dùng")

    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()
