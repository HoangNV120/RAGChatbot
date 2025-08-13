import pandas as pd
import numpy as np
import asyncio
import time
import json
from typing import List, Dict, Any
from datetime import datetime
import logging

# RAGAS imports - updated for latest version
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_similarity
    )
    from datasets import Dataset
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    print("✅ RAGAS imports successful")
except ImportError as e:
    print(f"RAGAS import error: {e}")
    print("Please install latest RAGAS: pip install ragas --upgrade")
    print("Also install: pip install datasets")

# Additional required imports for Excel export
try:
    import openpyxl
    print("✅ openpyxl available for Excel export")
except ImportError:
    print("⚠️ Installing openpyxl for Excel export...")
    import subprocess
    subprocess.run(["pip", "install", "openpyxl"])
    import openpyxl

# Internal imports
from app.master_chatbot import MasterChatbot
from app.document_processor import DocumentProcessor
from app.vector_store_small import VectorStoreSmall
from app.MultiModelChatAPI import MultiModelChatAPI
from app.config import settings
from ragas_config import RAGAS_METRICS, EVAL_SETTINGS, QUALITY_THRESHOLDS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGASEvaluator:
    def __init__(self):
        """
        Khởi tạo RAGAS Evaluator với MultiModelAPI
        """
        # Initialize MasterChatbot components
        self.master_chatbot = None
        self.doc_processor = None
        self.vector_store_small = None

        # Initialize MultiModelAPI for RAGAS evaluation
        self.multi_model_api = MultiModelChatAPI(
            api_key=settings.multi_model_api_key,
            model_name="gpt-4o-mini",  # Sử dụng model tốt cho evaluation
            api_url=settings.multi_model_api_url,
        )

        # Wrap MultiModelAPI cho RAGAS
        self.llm_wrapper = LangchainLLMWrapper(self.multi_model_api)

        # Results storage
        self.results = []
        self.latency_results = []

    async def initialize_chatbot(self):
        """
        Khởi tạo MasterChatbot với dual vector stores
        """
        try:
            print("🚀 Initializing RAG Chatbot components...")

            # Initialize Document processor
            import os
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "data")
            self.doc_processor = DocumentProcessor(data_dir=data_dir)

            # Initialize VectorStoreSmall
            self.vector_store_small = VectorStoreSmall()

            # Initialize Master Chatbot
            self.master_chatbot = MasterChatbot(
                vector_store=self.doc_processor.vector_store,
                vector_store_small=self.vector_store_small
            )

            print("✅ MasterChatbot initialized successfully!")

        except Exception as e:
            logger.error(f"Error initializing chatbot: {e}")
            raise

    def load_testcase_data(self, file_path: str) -> pd.DataFrame:
        """
        Đọc dữ liệu từ file Excel testcase
        """
        try:
            df = pd.read_excel(file_path)
            print(f"📊 Loaded {len(df)} test cases from {file_path}")
            print(f"Columns: {df.columns.tolist()}")

            # Kiểm tra các cột cần thiết
            required_columns = ['question', 'answer']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                # Thử các tên cột khác
                column_mapping = {
                    'Question': 'question',
                    'Q': 'question',
                    'query': 'question',
                    'Answer': 'answer',
                    'A': 'answer',
                    'response': 'answer',
                    'expected_answer': 'answer'
                }

                for old_col, new_col in column_mapping.items():
                    if old_col in df.columns and new_col in missing_columns:
                        df = df.rename(columns={old_col: new_col})
                        missing_columns.remove(new_col)
                        print(f"✅ Mapped column '{old_col}' to '{new_col}'")

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Lọc bỏ các dòng trống
            df = df.dropna(subset=['question', 'answer'])
            print(f"📝 After filtering: {len(df)} valid test cases")

            return df

        except Exception as e:
            logger.error(f"Error loading testcase data: {e}")
            raise

    async def generate_rag_response(self, question: str, session_id: str = None) -> Dict:
        """
        Sinh response từ RAG system và đo latency, đồng thời lấy contexts thực từ retrieval
        """
        start_time = time.time()

        try:
            # Generate response using MasterChatbot
            response = await self.master_chatbot.generate_response(
                query=question,
                session_id=session_id or f"eval_{int(time.time())}"
            )

            end_time = time.time()
            latency = end_time - start_time

            # Lấy contexts thực từ retrieval process
            contexts = await self.extract_real_contexts_from_rag(question, response)

            return {
                'answer': response.get('output', ''),
                'latency': latency,
                'route_used': response.get('route_used', 'UNKNOWN'),
                'session_id': response.get('session_id', ''),
                'contexts': contexts
            }

        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            end_time = time.time()
            latency = end_time - start_time

            return {
                'answer': f"Error: {str(e)}",
                'latency': latency,
                'route_used': 'ERROR',
                'session_id': '',
                'contexts': [f"Error retrieving context: {str(e)}"]
            }

    async def extract_real_contexts_from_rag(self, question: str, response: Dict) -> List[str]:
        """
        Lấy contexts thực từ RAG retrieval process - sửa để lấy đúng từ vector store tương ứng
        """
        contexts = []

        try:
            route_used = response.get('route_used', '')

            if route_used == 'RAGSMALL_MATCH':
                # Nếu dùng ragsmall, lấy context từ vector_store_small như trong master_chatbot
                print(f"   📖 Getting contexts from vector_store_small for RAGSMALL route")

                ragsmall_info = response.get('ragsmall_info', {})

                # Lấy contexts từ vector_store_small search (giống như master_chatbot)
                try:
                    ragsmall_results = await self.vector_store_small.similarity_search_with_score(
                        query=question,
                        k=3  # Lấy top 3 để có đủ context
                    )

                    for doc, score in ragsmall_results:

                        # Thêm answer từ metadata làm context
                        if 'answer' in doc.metadata:
                            contexts.append(doc.metadata['answer'])

                        # Limit contexts
                        if len(contexts) >= 4:
                            break

                except Exception as e:
                    logger.warning(f"Error getting contexts from vector_store_small: {e}")
                    # Fallback: sử dụng info từ response
                    if 'matched_question' in ragsmall_info:
                        contexts.append(ragsmall_info['matched_question'])
                    if 'matched_category' in ragsmall_info:
                        contexts.append(f"Category: {ragsmall_info['matched_category']}")

            elif 'RAG_CHAT' in route_used:
                # Nếu dùng RAG_CHAT, lấy contexts từ vector_store chính
                print(f"   📖 Getting contexts from main vector_store for RAG_CHAT route")
                contexts = await self.get_rag_chat_contexts(question)

            else:
                # Fallback: thực hiện retrieval để lấy contexts từ main vector store
                print(f"   📖 Getting contexts from main vector_store for {route_used} route")
                contexts = await self.get_rag_chat_contexts(question)

        except Exception as e:
            logger.warning(f"Error extracting real contexts: {e}")
            # Fallback: lấy contexts bằng cách search trực tiếp từ main vector store
            contexts = await self.get_rag_chat_contexts(question)

        # Đảm bảo có ít nhất 1 context
        if not contexts:
            contexts = [f"No relevant context found for: {question[:100]}..."]

        print(f"   📖 Total contexts extracted: {len(contexts)}")
        return contexts

    async def get_rag_chat_contexts(self, question: str) -> List[str]:
        """
        Lấy contexts thực từ RAG chat retrieval process
        """
        contexts = []

        try:
            # Sử dụng query rewriter như trong rag_chat
            rewrite_result = await self.master_chatbot.rag_chat.query_rewriter.analyze_and_rewrite(question)
            processed_query = rewrite_result.get("rewritten_query", question)
            subqueries = rewrite_result.get("subqueries", [processed_query])

            # Thực hiện retrieval như trong rag_chat
            k_per_query = max(1, min(3, 6 // len(subqueries)))

            # Sử dụng batch search như trong rag_chat
            all_results = await self.master_chatbot.rag_chat.vector_store.batch_similarity_search(
                subqueries,
                k=k_per_query
            )

            # Kết hợp và loại bỏ trùng lặp như trong rag_chat
            seen_content = set()
            for results in all_results:
                if isinstance(results, list):
                    for doc in results:
                        content_hash = hash(doc.page_content[:200])
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            contexts.append(doc.page_content)

                            # Limit to 3-4 contexts để tránh quá dài
                            if len(contexts) >= 4:
                                break

        except Exception as e:
            logger.warning(f"Error in RAG chat contexts retrieval: {e}")

            # Fallback: sử dụng similarity search đơn giản
            try:
                docs = await self.master_chatbot.rag_chat.vector_store.similarity_search(question, k=3)
                contexts = [doc.page_content for doc in docs]
            except Exception as fallback_e:
                logger.error(f"Fallback context retrieval failed: {fallback_e}")
                contexts = [f"Context retrieval failed: {str(fallback_e)}"]

        return contexts

    async def evaluate_single_question(self, question: str, expected_answer: str, index: int) -> Dict:
        """
        Đánh giá một câu hỏi đơn lẻ với RAGAS
        """
        print(f"\n🔍 Evaluating question {index + 1}: {question[:50]}...")

        # Generate RAG response
        rag_result = await self.generate_rag_response(question, f"eval_session_{index}")

        # Prepare data for RAGAS evaluation
        eval_data = {
            'question': [question],
            'answer': [rag_result['answer']],
            'ground_truth': [expected_answer],
            'contexts': [rag_result['contexts']]
        }

        result = {
            'question_index': index + 1,
            'question': question,
            'expected_answer': expected_answer,
            'generated_answer': rag_result['answer'],
            'latency': rag_result['latency'],
            'route_used': rag_result['route_used'],
            'contexts': rag_result['contexts']
        }

        # Evaluate with RAGAS
        try:
            # Convert to Dataset
            dataset = Dataset.from_dict(eval_data)

            # Wrap embeddings
            embeddings_wrapper = LangchainEmbeddingsWrapper(
                self.doc_processor.vector_store.embeddings
            )

            # Run evaluation
            evaluation_result = evaluate(
                dataset=dataset,
                metrics=RAGAS_METRICS,
                llm=self.llm_wrapper,
                embeddings=embeddings_wrapper
            )

            # Extract scores - chỉ 2 metrics: answer_similarity, faithfulness
            if hasattr(evaluation_result, 'to_pandas'):
                eval_df = evaluation_result.to_pandas()
                print(eval_df)
                if len(eval_df) > 0:
                    first_row = eval_df.iloc[0]
                    result.update({
                        'answer_similarity': float(first_row.get('semantic_similarity', 0.0)),
                        'faithfulness': float(first_row.get('faithfulness', 0.0))
                    })
                else:
                    # Set default values if no results
                    result.update({
                        'answer_similarity': 0.0,
                        'faithfulness': 0.0
                    })
            else:
                # Fallback for different RAGAS return format
                result.update({
                    'answer_similarity': float(evaluation_result.get('semantic_similarity', 0.0)),
                    'faithfulness': float(evaluation_result.get('faithfulness', 0.0))
                })

            print(f"✅ Question {index + 1} evaluated successfully")
            print(f"   Latency: {rag_result['latency']:.3f}s")
            print(f"   Route: {rag_result['route_used']}")
            print(f"   Answer Similarity: {result.get('answer_similarity', 0):.3f}")
            print(f"   Faithfulness: {result.get('faithfulness', 0):.3f}")

        except Exception as e:
            logger.error(f"Error in RAGAS evaluation for question {index + 1}: {e}")
            result.update({
                'answer_similarity': 0.0,
                'faithfulness': 0.0,
                'evaluation_error': str(e)
            })

        return result

    async def run_evaluation(self, testcase_file: str = "testcase.xlsx"):
        """
        Chạy đánh giá toàn bộ testcase
        """
        print("🚀 Starting RAGAS Evaluation...")

        # Initialize chatbot
        await self.initialize_chatbot()

        # Load test data
        df = self.load_testcase_data(testcase_file)

        # Run evaluation for each question
        evaluation_results = []

        for index, row in df.iterrows():
            question = str(row['question']).strip()
            expected_answer = str(row['answer']).strip()

            if not question or question.lower() in ['nan', 'none', '']:
                print(f"⚠️ Skipping empty question at index {index}")
                continue

            result = await self.evaluate_single_question(question, expected_answer, index)
            evaluation_results.append(result)

            # Store latency for statistics
            self.latency_results.append(result['latency'])

            # Delay to avoid overwhelming the system
            await asyncio.sleep(EVAL_SETTINGS['delay_between_requests'])

        self.results = evaluation_results
        return evaluation_results

    def calculate_statistics(self) -> Dict:
        """
        Tính toán thống kê tổng hợp - chỉ 2 metrics: answer_similarity, faithfulness
        """
        if not self.results:
            return {}

        # Extract numeric metrics - chỉ 3 metrics chính (sửa tên metric đúng)
        metrics = ['answer_similarity', 'faithfulness']

        stats = {
            'total_questions': len(self.results),
            'successful_evaluations': len([r for r in self.results if 'evaluation_error' not in r]),
            'average_metrics': {},
            'latency_statistics': {},
            'quality_assessment': {}
        }

        # Calculate average metrics - INCLUDE zeros trong calculation
        for metric in metrics:
            # Lấy tất cả values bao gồm cả 0 (nhưng exclude NaN và null)
            values = [r.get(metric, 0) for r in self.results if isinstance(r.get(metric), (int, float))]

            if values:
                mean_val = np.mean(values)
                stats['average_metrics'][metric] = {
                    'mean': mean_val,
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values),
                    'zero_count': len([v for v in values if v == 0.0]),  # Đếm số lượng 0
                    'non_zero_count': len([v for v in values if v > 0.0])  # Đếm số lượng > 0
                }

                # Quality assessment
                thresholds = QUALITY_THRESHOLDS.get(metric, {})
                if mean_val >= thresholds.get('excellent', 0.8):
                    quality = 'Excellent'
                elif mean_val >= thresholds.get('good', 0.6):
                    quality = 'Good'
                elif mean_val >= thresholds.get('acceptable', 0.4):
                    quality = 'Acceptable'
                else:
                    quality = 'Needs Improvement'

                stats['quality_assessment'][metric] = quality
            else:
                # Trường hợp không có values nào
                stats['average_metrics'][metric] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'count': 0,
                    'zero_count': 0,
                    'non_zero_count': 0
                }
                stats['quality_assessment'][metric] = 'No Data'

        # Calculate latency statistics
        if self.latency_results:
            stats['latency_statistics'] = {
                'mean': np.mean(self.latency_results),
                'std': np.std(self.latency_results),
                'min': np.min(self.latency_results),
                'max': np.max(self.latency_results),
                'p50': np.percentile(self.latency_results, 50),
                'p99': np.percentile(self.latency_results, 99)
            }

        return stats

    def export_results(self, output_file: str = None):
        """
        Xuất kết quả ra file Excel
        """
        if not self.results:
            print("❌ No results to export")
            return

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"ragas_evaluation_results_{timestamp}.xlsx"

        # Create DataFrame from results
        df_results = pd.DataFrame(self.results)

        # Calculate statistics
        stats = self.calculate_statistics()

        # Create statistics DataFrame
        stats_data = []

        # Add average metrics
        for metric, values in stats.get('average_metrics', {}).items():
            stats_data.append({
                'Metric': f"{metric}_mean",
                'Value': values['mean'],
                'Quality': stats['quality_assessment'].get(metric, 'N/A')
            })
            stats_data.append({
                'Metric': f"{metric}_std",
                'Value': values['std'],
                'Quality': ''
            })

        # Add latency statistics
        for stat_name, value in stats.get('latency_statistics', {}).items():
            stats_data.append({
                'Metric': f"latency_{stat_name}",
                'Value': value,
                'Quality': ''
            })

        df_stats = pd.DataFrame(stats_data)

        # Create summary info
        summary_info = [
            {'Info': 'Total Questions', 'Value': stats['total_questions']},
            {'Info': 'Successful Evaluations', 'Value': stats['successful_evaluations']},
            {'Info': 'Success Rate', 'Value': f"{(stats['successful_evaluations']/stats['total_questions']*100):.1f}%"},
            {'Info': 'Evaluation Date', 'Value': datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {'Info': 'Model Used', 'Value': 'gpt-4o-mini (via MultiModelAPI)'},
            {'Info': 'RAG System', 'Value': 'MasterChatbot with Dual Vector Stores'}
        ]
        df_summary = pd.DataFrame(summary_info)

        # Export to Excel with multiple sheets
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='Detailed_Results', index=False)
            df_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            df_summary.to_excel(writer, sheet_name='Evaluation_Info', index=False)

        print(f"✅ Results exported to: {output_file}")

        # Print summary
        print("\n📊 EVALUATION SUMMARY:")
        print(f"Total Questions: {stats['total_questions']}")
        print(f"Successful Evaluations: {stats['successful_evaluations']}")
        print(f"Success Rate: {(stats['successful_evaluations']/stats['total_questions']*100):.1f}%")

        if 'average_metrics' in stats:
            print("\n📈 Average Metrics:")
            for metric, values in stats['average_metrics'].items():
                quality = stats['quality_assessment'].get(metric, 'N/A')
                print(f"  {metric}: {values['mean']:.4f} (±{values['std']:.4f}) - {quality}")

        if 'latency_statistics' in stats:
            print(f"\n⏱️ Latency Statistics:")
            latency_stats = stats['latency_statistics']
            print(f"  Mean: {latency_stats['mean']:.3f}s")
            print(f"  P50: {latency_stats['p50']:.3f}s")
            print(f"  P99: {latency_stats['p99']:.3f}s")
            print(f"  Min: {latency_stats['min']:.3f}s")
            print(f"  Max: {latency_stats['max']:.3f}s")

async def main():
    """
    Main function để chạy evaluation
    """
    evaluator = RAGASEvaluator()

    try:
        # Run evaluation
        results = await evaluator.run_evaluation("testcase.xlsx")

        # Export results
        evaluator.export_results()

        print("\n🎉 Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Error in main evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
