"""
RAGAS Evaluation Configuration
Cấu hình các metrics và settings cho đánh giá RAG system
"""

from ragas.metrics import (
    faithfulness,
    answer_similarity
)

# RAGAS Metrics Configuration - Chỉ 3 metrics chính
RAGAS_METRICS = [
    answer_similarity,      # Đo độ tương đồng ngữ nghĩa với ground truth
    faithfulness,           # Đo độ trung thực của câu trả lời dựa trên context      # Đo độ liên quan của câu trả lời với câu hỏi
]

# Evaluation settings
EVAL_SETTINGS = {
    'batch_size': 1,          # Process one by one để tránh rate limit
    'delay_between_requests': 0.5,  # Delay giữa các request (seconds)
    'max_retries': 3,         # Số lần retry khi có lỗi
    'timeout': 30,            # Timeout cho mỗi request (seconds)
}

# Output settings
OUTPUT_SETTINGS = {
    'include_detailed_results': True,
    'include_summary_stats': True,
    'include_latency_breakdown': True,
    'export_format': 'xlsx'
}

# Metrics thresholds cho evaluation quality - Chỉ 3 metrics
QUALITY_THRESHOLDS = {
    'answer_similarity': {
        'excellent': 0.8,
        'good': 0.6,
        'acceptable': 0.4
    },
    'faithfulness': {
        'excellent': 0.8,
        'good': 0.6,
        'acceptable': 0.4
    },
}
