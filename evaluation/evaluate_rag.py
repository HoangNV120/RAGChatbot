"""
Đánh giá RAG Chat sử dụng RAGAS
Đánh giá các tiêu chí: Faithfulness, Answer Relevancy, Answer Correctness
"""

import asyncio
import pandas as pd
from typing import List, Dict, Any
import os
import sys
from dotenv import load_dotenv
import json

# Thêm thư mục gốc vào Python path để import các module RAG
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RAGAS
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity,

    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
    AnswerSimilarity,
)
# Import LLM configuration
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Import các module RAG
from app.rag_chat import RAGChat
from app.vector_store import VectorStore
from app.document_processor import DocumentProcessor
from app.safety_guard import SafetyGuard
from app.config import settings

load_dotenv()

class RAGEvaluator:
    def __init__(self):
        """Khởi tạo evaluator với RAG components"""
        self.rag_chat = None
        self.vector_store = None
        self.doc_processor = None
        self.safety_guard = None
        self.llm = None
        self.embeddings = None

        # Khởi tạo LLM và embeddings từ config
        self._setup_llm_and_embeddings()

    def _setup_llm_and_embeddings(self):
        """Thiết lập LLM và embeddings từ config"""
        try:
            # Khởi tạo LLM từ config settings
            self.llm = ChatOpenAI(
                model=settings.model_name,
                temperature=settings.temperature,
                api_key=settings.openai_api_key
            )

            # Khởi tạo embeddings từ config settings
            self.embeddings = OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.openai_api_key
            )

            print(f"✅ Đã thiết lập LLM: {settings.model_name}")
            print(f"✅ Đã thiết lập Embeddings: {settings.embedding_model}")

        except Exception as e:
            print(f"❌ Lỗi khi thiết lập LLM/Embeddings: {e}")
            raise

    async def setup_rag_components(self):
        """Khởi tạo các thành phần RAG"""
        print("🚀 Khởi tạo các thành phần RAG...")

        # Khởi tạo document processor - sửa đường dẫn từ evaluation folder
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "data")
        self.doc_processor = DocumentProcessor(data_dir=data_dir)

        # Khởi tạo RAG chat
        self.rag_chat = RAGChat(vector_store=self.doc_processor.vector_store)

        # # Khởi tạo Safety Guard
        # self.safety_guard = SafetyGuard()
        # print("✅ Safety Guard đã được khởi tạo")

        # Load documents nếu chưa load
        await self.doc_processor.load_and_process_all()
        print("✅ RAG components đã sẵn sàng!")

    def load_test_data(self, excel_path: str) -> pd.DataFrame:
        """
        Load dữ liệu test từ file Excel
        Expected format: columns ['question', 'answer']
        """
        try:
            df = pd.read_excel(excel_path)
            print(f"📊 Đã load {len(df)} test cases từ {excel_path}")

            # Kiểm tra format
            required_columns = ['question', 'answer']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            return df

        except Exception as e:
            print(f"❌ Lỗi khi load file Excel: {e}")
            raise

    async def get_rag_response_with_context(self, question: str) -> Dict[str, Any]:
        """
        Lấy response và context từ RAG system với Safety Guard
        """
        try:
            # # 🛡️ Kiểm tra Safety Guard trước khi vào RAG
            # print(f"🔍 Kiểm tra safety cho câu hỏi: {question[:50]}...")
            # safety_result = await self.safety_guard.check_safety(question)
            #
            # if not safety_result["is_safe"]:
            #     # Nếu không an toàn, trả về response từ safety guard
            #     safety_reason = safety_result["reason"]
            #     # Đảm bảo safety_reason là string, không phải None hoặc kiểu khác
            #     if safety_reason is None:
            #         safety_reason = "Câu hỏi không phù hợp với hệ thống."
            #     elif not isinstance(safety_reason, str):
            #         safety_reason = str(safety_reason)
            #
            #     print(f"⚠️ Câu hỏi bị từ chối bởi Safety Guard: {safety_reason}")
            #     return {
            #         "contexts": ["[SAFETY BLOCKED] " + str(safety_reason)],  # luôn là list[str]
            #         "answer": str(safety_reason),                             # luôn là string
            #         "safety_status": "blocked"
            #     }
            #
            # print("✅ Câu hỏi đã qua kiểm tra safety, tiếp tục với RAG...")

            # Lấy documents liên quan (context)
            docs = await self.rag_chat.vector_store.similarity_search(question, k=4)

            # Tạo context từ documents
            contexts = []
            if docs:
                for doc in docs:
                    contexts.append(doc.page_content)
            else:
                contexts = ["Không tìm thấy thông tin liên quan."]

            # Lấy response từ RAG
            rag_result = await self.rag_chat.generate_response(question)
            actual_output = rag_result.get("output", "")

            # Đảm bảo actual_output là string
            if not isinstance(actual_output, str):
                actual_output = str(actual_output)

            return {
                "contexts": contexts,  # List of strings
                "answer": actual_output,  # String
                "safety_status": "approved"
            }

        except Exception as e:
            print(f"❌ Lỗi khi lấy RAG response: {e}")
            return {
                "contexts": ["Đã xảy ra lỗi trong quá trình xử lý"],  # List of strings
                "answer": f"Lỗi: {str(e)}",  # String
                "safety_status": "error"
            }

    async def create_evaluation_dataset(self, df: pd.DataFrame) -> List[SingleTurnSample]:
        """
        Tạo dataset cho RAGAS từ DataFrame
        """
        samples = []

        print("🔄 Tạo evaluation dataset...")
        for idx, row in df.iterrows():
            question = str(row['question'])
            ground_truth = str(row['answer'])

            print(f"Xử lý test case {idx + 1}/{len(df)}: {question[:50]}...")

            # Lấy response và context từ RAG
            rag_result = await self.get_rag_response_with_context(question)

            # Tạo sample theo format RAGAS SingleTurnSample object
            sample = SingleTurnSample(
                user_input=question,
                response=rag_result["answer"],
                retrieved_contexts=rag_result["contexts"],
                reference=ground_truth
            )
            samples.append(sample)

        print(f"✅ Đã tạo dataset với {len(samples)} test cases")

        # Trả về list các SingleTurnSample objects cho RAGAS
        return samples

    def setup_metrics(self) -> List:
        """
        Thiết lập các metrics để đánh giá với RAGAS sử dụng LLM từ config
        """
        # Cấu hình metrics với LLM từ config
        print(f"🔧 Cấu hình RAGAS metrics với LLM: {settings.model_name}")

        # Import để cấu hình LLM cho RAGAS
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        # Tạo RAGAS LLM và embeddings wrappers
        ragas_llm = LangchainLLMWrapper(self.llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)

        # Cấu hình metrics với LLM tùy chỉnh
        metrics = [
            faithfulness,
            answer_correctness,
            answer_similarity
        ]

        print(f"✅ Đã cấu hình {len(metrics)} metrics với LLM tùy chỉnh")
        return metrics

    async def run_evaluation(self, excel_path: str, output_path: str = "ragas_evaluation_results.json"):
        """
        Chạy đánh giá hoàn chỉnh với RAGAS
        """
        try:
            # Setup RAG components
            await self.setup_rag_components()

            # Load test data
            df = self.load_test_data(excel_path)

            # Create evaluation dataset - nhận list of samples
            samples = await self.create_evaluation_dataset(df)

            # Convert to Dataset sử dụng constructor với samples
            dataset = EvaluationDataset(samples=samples)

            # Setup metrics
            metrics = self.setup_metrics()

            print("🎯 Bắt đầu đánh giá với RAGAS...")
            print(f"🤖 Sử dụng LLM: {settings.model_name}")
            print(f"🔍 Sử dụng Embeddings: {settings.embedding_model}")

            # Run evaluation
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings,
            )

            # Save results - tạo dataset_dict để tương thích với save method
            dataset_dict = {
                "question": [sample.user_input for sample in samples],
                "answer": [sample.response for sample in samples],
                "contexts": [sample.retrieved_contexts for sample in samples],
                "ground_truth": [sample.reference for sample in samples]
            }

            self.save_detailed_results(result, dataset_dict, output_path)

            print(f"✅ Đánh giá hoàn thành! Kết quả được lưu tại: {output_path}")

            return result

        except Exception as e:
            print(f"❌ Lỗi trong quá trình đánh giá: {e}")
            raise

    def save_detailed_results(self, evaluation_result, dataset_dict: Dict, output_path: str):
        """
        Lưu kết quả chi tiết vào file
        """
        # Debug: In ra cấu trúc của evaluation_result để hiểu rõ hơn
        print(f"🔍 Debug - Type của evaluation_result: {type(evaluation_result)}")

        # Tạo kết quả chi tiết cho từng test case
        detailed_results = []

        for i in range(len(dataset_dict["question"])):
            result = {
                "test_case_id": i + 1,
                "question": dataset_dict["question"][i],
                "answer": dataset_dict["answer"][i],
                "ground_truth": dataset_dict["ground_truth"][i],
                "contexts": dataset_dict["contexts"][i],
                "metrics_scores": {}
            }

            # Thử các cách khác nhau để truy cập scores
            try:
                # Cách 1: Nếu là DataFrame (phổ biến với RAGAS mới)
                if hasattr(evaluation_result, 'to_pandas'):
                    df = evaluation_result.to_pandas()
                    if i < len(df):
                        for col in df.columns:
                            if col not in ['question', 'answer', 'contexts', 'ground_truth', 'user_input', 'retrieved_contexts', 'response', 'reference']:
                                result["metrics_scores"][col] = float(df.iloc[i][col]) if pd.notna(df.iloc[i][col]) else 0.0

                # Cách 2: Nếu có attribute scores
                elif hasattr(evaluation_result, 'scores'):
                    scores_dict = evaluation_result.scores
                    for metric_name, score_list in scores_dict.items():
                        if isinstance(score_list, list) and i < len(score_list):
                            result["metrics_scores"][metric_name] = float(score_list[i]) if score_list[i] is not None else 0.0
                        else:
                            result["metrics_scores"][metric_name] = float(score_list) if score_list is not None else 0.0

                # Cách 3: Nếu là dictionary trực tiếp (trường hợp này)
                elif isinstance(evaluation_result, dict):
                    for metric_name, score_list in evaluation_result.items():
                        if isinstance(score_list, list) and i < len(score_list):
                            result["metrics_scores"][metric_name] = float(score_list[i]) if score_list[i] is not None else 0.0
                        else:
                            result["metrics_scores"][metric_name] = float(score_list) if score_list is not None else 0.0

                # Cách 4: Truy cập trực tiếp các attributes
                else:
                    for metric_name in ['faithfulness', 'answer_relevancy', 'answer_correctness',
                                      'context_precision', 'context_recall', 'answer_similarity']:
                        if hasattr(evaluation_result, metric_name):
                            score_data = getattr(evaluation_result, metric_name)
                            if isinstance(score_data, list) and i < len(score_data):
                                result["metrics_scores"][metric_name] = float(score_data[i]) if score_data[i] is not None else 0.0
                            else:
                                result["metrics_scores"][metric_name] = float(score_data) if score_data is not None else 0.0

            except Exception as e:
                print(f"⚠️ Lỗi khi truy cập scores cho test case {i+1}: {e}")

            detailed_results.append(result)

        # Convert evaluation_result để lấy raw data trước
        raw_data = self._convert_evaluation_results_to_dict(evaluation_result)

        # Nếu metrics_scores vẫn trống, thử extract từ raw data
        if detailed_results and not detailed_results[0]["metrics_scores"]:
            print("🔄 Metrics scores trống, thử extract từ raw data...")
            if isinstance(raw_data, list):
                for i, raw_item in enumerate(raw_data):
                    if i < len(detailed_results):
                        # Extract scores từ raw data
                        for metric_name in ['faithfulness', 'answer_relevancy', 'answer_correctness',
                                          'context_precision', 'context_recall', 'answer_similarity']:
                            if metric_name in raw_item:
                                score = raw_item[metric_name]
                                detailed_results[i]["metrics_scores"][metric_name] = float(score) if score is not None else 0.0

        # Tạo summary
        summary = {
            "total_test_cases": len(dataset_dict["question"]),
            "average_scores": {},
            "overall_performance": {},
            "evaluation_config": {
                "llm_model": settings.model_name,
                "embedding_model": settings.embedding_model,
                "temperature": settings.temperature
            }
        }

        # Tính điểm trung bình cho từng metric
        all_metrics = set()
        for detail_result in detailed_results:
            all_metrics.update(detail_result["metrics_scores"].keys())

        for metric_name in all_metrics:
            scores = []
            for detail_result in detailed_results:
                if metric_name in detail_result["metrics_scores"]:
                    scores.append(detail_result["metrics_scores"][metric_name])

            avg_score = sum(scores) / len(scores) if scores else 0
            summary["average_scores"][metric_name] = avg_score

        # Đánh giá tổng thể
        avg_faithfulness = summary["average_scores"].get("faithfulness", 0)
        avg_relevancy = summary["average_scores"].get("answer_relevancy", 0)
        avg_correctness = summary["average_scores"].get("answer_correctness", 0)

        overall_score = (avg_faithfulness + avg_relevancy + avg_correctness) / 3
        summary["overall_performance"]["overall_score"] = overall_score

        if overall_score >= 0.8:
            summary["overall_performance"]["grade"] = "Excellent"
        elif overall_score >= 0.7:
            summary["overall_performance"]["grade"] = "Good"
        elif overall_score >= 0.6:
            summary["overall_performance"]["grade"] = "Fair"
        else:
            summary["overall_performance"]["grade"] = "Needs Improvement"

        # Lưu kết quả
        final_results = {
            "summary": summary,
            "detailed_results": detailed_results,
            "raw_evaluation_result": raw_data
        }

        # Lưu vào file JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

        # Tạo báo cáo tóm tắt
        self.create_summary_report(summary, detailed_results, output_path.replace('.json', '_summary.txt'))

        # Debug: In ra một số kết quả để kiểm tra
        print(f"🔍 Debug - Số metrics tìm được: {len(all_metrics)}")
        print(f"🔍 Debug - Metrics: {list(all_metrics)}")
        if detailed_results:
            print(f"🔍 Debug - Scores của test case đầu tiên: {detailed_results[0]['metrics_scores']}")

    def _convert_evaluation_results_to_dict(self, evaluation_result):
        """
        Chuyển đổi evaluation result thành dictionary để lưu vào JSON
        """
        try:
            # Thử convert pandas DataFrame
            if hasattr(evaluation_result, 'to_pandas'):
                df = evaluation_result.to_pandas()
                return df.to_dict('records')

            # Thử convert dictionary
            elif isinstance(evaluation_result, dict):
                result_dict = {}
                for key, value in evaluation_result.items():
                    if hasattr(value, 'tolist'):
                        result_dict[key] = value.tolist()
                    elif isinstance(value, (int, float, str, bool, list, dict)):
                        result_dict[key] = value
                    else:
                        result_dict[key] = str(value)
                return result_dict

            # Fallback: truy cập attributes
            else:
                result_dict = {}
                for attr in dir(evaluation_result):
                    if not attr.startswith('_') and not callable(getattr(evaluation_result, attr)):
                        try:
                            value = getattr(evaluation_result, attr)
                            if hasattr(value, 'tolist'):
                                result_dict[attr] = value.tolist()
                            elif isinstance(value, (int, float, str, bool, list, dict)):
                                result_dict[attr] = value
                            else:
                                result_dict[attr] = str(value)
                        except:
                            continue
                return result_dict

        except Exception as e:
            print(f"⚠️ Không thể chuyển đổi evaluation result: {e}")
            return {"error": str(e), "type": str(type(evaluation_result))}

    def create_summary_report(self, summary: Dict, detailed_results: List[Dict], summary_path: str):
        """
        Tạo báo cáo tổng kết dạng text
        """
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("BÁO CÁO ĐÁNH GIÁ RAG SYSTEM VỚI RAGAS\n")
            f.write("="*60 + "\n\n")

            # Thông tin tổng quan
            f.write(f"Tổng số test cases: {summary['total_test_cases']}\n")
            f.write(f"Điểm tổng thể: {summary['overall_performance']['overall_score']:.3f}\n")
            f.write(f"Xếp loại: {summary['overall_performance']['grade']}\n\n")

            # Điểm trung bình các metrics
            f.write("ĐIỂM TRUNG BÌNH CÁC METRICS:\n")
            f.write("-"*40 + "\n")
            for metric_name, score in summary['average_scores'].items():
                f.write(f"{metric_name:20}: {score:.3f}\n")

            f.write("\n" + "="*60 + "\n")
            f.write("CHI TIẾT TỪNG TEST CASE:\n")
            f.write("="*60 + "\n\n")

            # Chi tiết từng test case
            for result in detailed_results:
                f.write(f"Test Case {result['test_case_id']}:\n")
                f.write(f"Question: {result['question'][:100]}...\n")
                f.write(f"Metrics scores:\n")
                for metric_name, score in result['metrics_scores'].items():
                    f.write(f"  - {metric_name}: {score:.3f}\n")
                f.write("\n")

        print(f"📄 Báo cáo tóm tắt được lưu tại: {summary_path}")

    def print_evaluation_summary(self, result):
        """
        In tóm tắt kết quả đánh giá ra console
        """
        print("\n" + "="*60)
        print("📊 KẾT QUẢ ĐÁNH GIÁ RAG SYSTEM VỚI RAGAS")
        print("="*60)

        try:
            # Thử các cách khác nhau để lấy scores
            scores_dict = {}

            # Cách 1: DataFrame
            if hasattr(result, 'to_pandas'):
                df = result.to_pandas()
                for col in df.columns:
                    if col not in ['question', 'answer', 'contexts', 'ground_truth', 'user_input', 'retrieved_contexts', 'response', 'reference']:
                        scores = df[col].dropna().tolist()
                        if scores:
                            avg_score = sum(scores) / len(scores)
                            scores_dict[col] = avg_score

            # Cách 2: Dictionary hoặc scores attribute
            elif hasattr(result, 'scores') or isinstance(result, dict):
                data = result.scores if hasattr(result, 'scores') else result
                for metric_name, score_data in data.items():
                    if isinstance(score_data, list):
                        valid_scores = [s for s in score_data if s is not None]
                        if valid_scores:
                            scores_dict[metric_name] = sum(valid_scores) / len(valid_scores)
                    else:
                        scores_dict[metric_name] = float(score_data) if score_data is not None else 0.0

            # Cách 3: Attributes trực tiếp
            else:
                for metric_name in ['faithfulness', 'answer_relevancy', 'answer_correctness',
                                  'context_precision', 'context_recall', 'semantic_similarity']:
                    if hasattr(result, metric_name):
                        score_data = getattr(result, metric_name)
                        if isinstance(score_data, list):
                            valid_scores = [s for s in score_data if s is not None]
                            if valid_scores:
                                scores_dict[metric_name] = sum(valid_scores) / len(valid_scores)
                        else:
                            scores_dict[metric_name] = float(score_data) if score_data is not None else 0.0

            # Nếu vẫn không có scores, thử convert và extract
            if not scores_dict:
                print("🔄 Không tìm thấy scores, thử convert result...")
                converted = self._convert_evaluation_results_to_dict(result)
                if isinstance(converted, list):
                    # Tính trung bình từ converted data
                    metrics_sums = {}
                    metrics_counts = {}

                    for item in converted:
                        for metric_name in ['faithfulness', 'answer_relevancy', 'answer_correctness',
                                          'context_precision', 'context_recall', 'semantic_similarity']:
                            if metric_name in item and item[metric_name] is not None:
                                if metric_name not in metrics_sums:
                                    metrics_sums[metric_name] = 0
                                    metrics_counts[metric_name] = 0
                                metrics_sums[metric_name] += float(item[metric_name])
                                metrics_counts[metric_name] += 1

                    for metric_name in metrics_sums:
                        if metrics_counts[metric_name] > 0:
                            scores_dict[metric_name] = metrics_sums[metric_name] / metrics_counts[metric_name]

            # In kết quả
            if scores_dict:
                for metric_name, avg_score in scores_dict.items():
                    print(f"{metric_name:20}: {avg_score:.3f}")
            else:
                print("⚠️ Không tìm thấy scores trong kết quả evaluation")
                print(f"🔍 Type: {type(result)}")
                print(f"🔍 Attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")

        except Exception as e:
            print(f"❌ Lỗi khi in summary: {e}")
            print(f"🔍 Type của result: {type(result)}")

        print("="*60)

async def main():
    """
    Hàm main để chạy evaluation
    """
    # Khởi tạo evaluator
    evaluator = RAGEvaluator()

    # Đường dẫn file Excel chứa test data
    excel_path = "../app/data_test.xlsx"  # Thay đổi đường dẫn nếu cần

    # Chạy evaluation
    try:
        results = await evaluator.run_evaluation(
            excel_path=excel_path,
            output_path="rag_evaluation_results.json"
        )
        print("🎉 Đánh giá hoàn thành thành công!")

    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    # Chạy evaluation
    asyncio.run(main())




