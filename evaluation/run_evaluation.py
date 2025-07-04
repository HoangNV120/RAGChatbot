"""
Script đơn giản để chạy RAG evaluation với RAGAS
"""
import asyncio
import sys
import os

# Thêm thư mục gốc vào Python path - sửa đường dẫn từ evaluation folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate_rag import RAGEvaluator

async def run_evaluation():
    """Chạy evaluation với RAGAS"""

    print("🚀 Bắt đầu đánh giá RAG Chat với RAGAS...")

    # Khởi tạo evaluator
    evaluator = RAGEvaluator()

    # Đường dẫn file Excel - sửa đường dẫn từ evaluation folder
    excel_path = "../app/data/data_evaluation.xlsx"  # File nằm trong app/data/

    # Kiểm tra file Excel có tồn tại không
    if not os.path.exists(excel_path):
        print(f"❌ Không tìm thấy file Excel: {excel_path}")
        print("📝 Vui lòng tạo file Excel với format:")
        print("   - Cột 'question': Câu hỏi test")
        print("   - Cột 'answer': Câu trả lời mong đợi (ground truth)")
        return

    try:
        # Chạy evaluation với RAGAS
        results = await evaluator.run_evaluation(
            excel_path=excel_path,
            output_path="ragas_evaluation_results.json"
        )

        # In kết quả tóm tắt
        evaluator.print_evaluation_summary(results)

        print("\n🎉 Evaluation với RAGAS hoàn thành!")
        print("📊 Kết quả được lưu tại:")
        print("  - ragas_evaluation_results.json (chi tiết)")
        print("  - ragas_evaluation_results_summary.txt (tóm tắt)")

    except Exception as e:
        print(f"❌ Lỗi trong quá trình evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Chạy evaluation
    asyncio.run(run_evaluation())
