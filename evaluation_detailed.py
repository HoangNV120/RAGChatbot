import pandas as pd
import asyncio
import time
import tiktoken
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
import logging
from typing import Dict, List, Any, Optional

# Import các components cần thiết
from app.master_chatbot import MasterChatbot
from app.vector_store import VectorStore
from app.vector_store_small import VectorStoreSmall
from app.config import settings

# Thiết lập logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class DetailedEvaluation:
    def __init__(self):
        """Khởi tạo evaluation system"""
        print("🚀 Initializing Detailed Evaluation System...")

        # Khởi tạo vector stores
        self.vector_store = VectorStore()
        self.vector_store_small = VectorStoreSmall()

        # Khởi tạo master chatbot
        self.master_chatbot = MasterChatbot(
            vector_store=self.vector_store,
            vector_store_small=self.vector_store_small
        )

        # Khởi tạo embedding model để tính similarity
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

        # Khởi tạo tiktoken encoder để tính cost
        self.encoder = tiktoken.encoding_for_model("gpt-4o-mini")

        # Pricing (USD per 1K tokens) - cập nhật theo giá mới nhất
        self.pricing = {
            "embedding_input": 0.00002,  # text-embedding-3-large
            "llm_input": 0.00015,        # gpt-4o-mini input
            "llm_output": 0.0006         # gpt-4o-mini output
        }

        print("✅ Evaluation system initialized successfully!")

    def count_tokens(self, text: str) -> int:
        """Đếm số tokens trong text"""
        if not text:
            return 0
        return len(self.encoder.encode(str(text)))

    def calculate_embedding_cost(self, text: str) -> float:
        """Tính cost cho embedding"""
        tokens = self.count_tokens(text)
        return (tokens / 1000) * self.pricing["embedding_input"]

    def calculate_llm_cost(self, input_text: str, output_text: str) -> tuple:
        """Tính cost cho LLM (input và output riêng biệt)"""
        input_tokens = self.count_tokens(input_text)
        output_tokens = self.count_tokens(output_text)

        input_cost = (input_tokens / 1000) * self.pricing["llm_input"]
        output_cost = (output_tokens / 1000) * self.pricing["llm_output"]

        return input_cost, output_cost, input_tokens, output_tokens

    async def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Tính cosine similarity giữa 2 text sử dụng embedding"""
        try:
            if not text1 or not text2:
                return 0.0

            # Tạo embeddings cho cả 2 text
            embeddings = await self.embeddings.aembed_documents([text1, text2])

            # Tính cosine similarity
            embedding1 = np.array(embeddings[0]).reshape(1, -1)
            embedding2 = np.array(embeddings[1]).reshape(1, -1)

            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

    async def evaluate_single_query(self, question: str, expected_answer: str, query_index: int) -> Dict[str, Any]:
        """Đánh giá một query duy nhất theo đúng luồng: Master chatbot → RAG_CHAT (nếu cần)"""
        print(f"\n{'='*60}")
        print(f"🔍 EVALUATING QUERY {query_index + 1}")
        print(f"Question: {question[:100]}...")
        print(f"{'='*60}")

        # Khởi tạo metrics
        metrics = {
            "question": question,
            "expected_answer": expected_answer,
            "query_index": query_index + 1,

            # Master chatbot phase (ragsmall)
            "master_chatbot_time": 0,
            "ragsmall_search_cost": 0,

            # RAG_CHAT phases (chỉ có nếu threshold < 0.8)
            "pre_retrieval_time": 0,
            "pre_retrieval_cost": 0,
            "pre_retrieval_tokens": 0,
            "embedding_time": 0,
            "embedding_cost": 0,  # ✅ Thêm embedding_cost với giá trị mặc định
            "vector_search_time": 0,
            "generation_time": 0,
            "generation_cost": 0,

            # Result metrics
            "actual_answer": "",
            "route_used": "",
            "chunks_retrieved": [],
            "ragsmall_similarity": 0,
            "cosine_similarity": 0,
            "accuracy_score": 0,
            "total_time": 0,
            "total_cost": 0,

            # Error tracking
            "errors": []
        }

        total_start_time = time.time()

        try:
            # === PHASE 1: MASTER CHATBOT (Full pipeline) ===
            print("🏠 Phase 1: Master Chatbot (full pipeline)...")
            master_start_time = time.time()

            # Chạy master chatbot - nó sẽ tự động handle ragsmall → RAG_CHAT fallback
            response = await self.master_chatbot.generate_response(question)

            master_end_time = time.time()
            metrics["master_chatbot_time"] = master_end_time - master_start_time
            metrics["actual_answer"] = response.get("output", "")
            metrics["route_used"] = response.get("route_used", "UNKNOWN")

            print(f"   ⏱️  Master chatbot time: {metrics['master_chatbot_time']:.3f}s")
            print(f"   🎯 Route used: {metrics['route_used']}")

            # Thu thập ragsmall info nếu có
            ragsmall_info = response.get("ragsmall_info", {})
            if ragsmall_info:
                metrics["ragsmall_similarity"] = ragsmall_info.get("similarity_score", 0)
                print(f"   📈 Ragsmall similarity: {metrics['ragsmall_similarity']:.4f}")

            # Tính ragsmall search cost (embedding cost luôn có)
            ragsmall_embedding_cost = self.calculate_embedding_cost(question)
            metrics["ragsmall_search_cost"] = ragsmall_embedding_cost
            print(f"   💰 Ragsmall search cost: ${ragsmall_embedding_cost:.6f}")

            # === THU THẬP METRICS TỪ RAG_CHAT (nếu có fallback) ===
            if "FALLBACK" in metrics["route_used"] or "RAG_CHAT" in metrics["route_used"]:
                print("🔄 Phase 2: Extracting RAG_CHAT metrics...")

                # Lấy pre-retrieval metrics từ response
                pre_retrieval_metrics = response.get("pre_retrieval_metrics", {})
                if pre_retrieval_metrics:
                    metrics["pre_retrieval_time"] = pre_retrieval_metrics.get("pre_retrieval_time", 0)
                    metrics["pre_retrieval_cost"] = pre_retrieval_metrics.get("pre_retrieval_cost", 0)
                    metrics["pre_retrieval_tokens"] = pre_retrieval_metrics.get("total_tokens", 0)

                    print(f"   🧠 Pre-retrieval time: {metrics['pre_retrieval_time']:.3f}s")
                    print(f"   💰 Pre-retrieval cost: ${metrics['pre_retrieval_cost']:.6f}")
                    print(f"   📊 Pre-retrieval tokens: {metrics['pre_retrieval_tokens']}")

                # Lấy vector metrics từ response (thay vì estimate)
                vector_metrics = response.get("vector_metrics", {})
                if vector_metrics:
                    metrics["embedding_time"] = vector_metrics.get("rag_embedding_time", 0)
                    metrics["vector_search_time"] = vector_metrics.get("rag_vector_search_time", 0)

                    print(f"   🔍 RAG embedding time: {metrics['embedding_time']:.3f}s")
                    print(f"   🔍 RAG vector search time: {metrics['vector_search_time']:.3f}s")

                    # Tính embedding cost từ subqueries
                    subqueries = response.get("subqueries", [question])
                    total_embedding_cost = 0
                    for subquery in subqueries:
                        total_embedding_cost += self.calculate_embedding_cost(subquery)

                    metrics["embedding_cost"] = total_embedding_cost
                    print(f"   💰 RAG embedding cost: ${total_embedding_cost:.6f} (for {len(subqueries)} subqueries)")

                    # Lấy generation metrics thực tế từ vector_metrics (đã được combine)
                    metrics["generation_time"] = vector_metrics.get("generation_time", 0)

                    # ✅ Ưu tiên sử dụng generation_cost từ vector_metrics nếu có
                    if vector_metrics.get("generation_cost", 0) > 0:
                        metrics["generation_cost"] = vector_metrics.get("generation_cost", 0)
                    else:
                        # Fallback: tính manual nếu vector_metrics không có generation_cost
                        if metrics["actual_answer"]:
                            subqueries = response.get("subqueries", [question])
                            estimated_context = " ".join([f"Context for: {sq[:100]}" for sq in subqueries])
                            full_prompt = f"System prompt + Context: {estimated_context} + Question: {question}"

                            gen_input_cost, gen_output_cost, _, _ = self.calculate_llm_cost(
                                full_prompt, metrics["actual_answer"]
                            )
                            metrics["generation_cost"] = gen_input_cost + gen_output_cost
                        else:
                            metrics["generation_cost"] = 0

                    generation_tokens = vector_metrics.get("total_tokens", 0)

                    print(f"   🤖 Generation time: {metrics['generation_time']:.3f}s")
                    print(f"   💰 Generation cost: ${metrics['generation_cost']:.6f}")
                    print(f"   📊 Generation tokens: {generation_tokens}")

                    # Lấy chunks thực tế từ retrieved_docs thay vì estimate
                    retrieved_docs = response.get("retrieved_docs", [])
                    chunks_info = []
                    for doc_info in retrieved_docs:
                        chunks_info.append({
                            "content": doc_info.get("content", ""),
                            "content_length": doc_info.get("content_length", 0),
                            "chunk_index": doc_info.get("chunk_index", 0),
                            "metadata": doc_info.get("metadata", {})
                        })

                    metrics["chunks_retrieved"] = chunks_info
                    print(f"   📄 Actual chunks retrieved: {len(chunks_info)}")

                    # Log chi tiết từng chunk
                    for i, chunk in enumerate(chunks_info[:3]):  # Chỉ log 3 chunks đầu
                        print(f"      Chunk {i+1}: {chunk['content'][:100]}... (length: {chunk['content_length']})")
                else:
                    # Fallback estimate nếu không có vector_metrics
                    remaining_time = metrics["master_chatbot_time"] - metrics.get("pre_retrieval_time", 0)
                    metrics["embedding_time"] = remaining_time * 0.3  # ~30% for embedding
                    metrics["vector_search_time"] = remaining_time * 0.4  # ~40% for search
                    metrics["generation_time"] = remaining_time * 0.3  # ~30% for generation

                    # Estimate embedding cost từ subqueries
                    subqueries = response.get("subqueries", [question])
                    total_embedding_cost = 0
                    for subquery in subqueries:
                        total_embedding_cost += self.calculate_embedding_cost(subquery)
                    metrics["embedding_cost"] = total_embedding_cost

                    # Estimate generation cost
                    if metrics["actual_answer"]:
                        estimated_context = " ".join([f"Context for: {sq[:100]}" for sq in subqueries])
                        full_prompt = f"System prompt + Context: {estimated_context} + Question: {question}"

                        gen_input_cost, gen_output_cost, _, _ = self.calculate_llm_cost(
                            full_prompt, metrics["actual_answer"]
                        )
                        metrics["generation_cost"] = gen_input_cost + gen_output_cost

                    print(f"   ⚠️  Using fallback estimates (no vector_metrics)")
                    print(f"   🔍 Estimated embedding time: {metrics['embedding_time']:.3f}s")
                    print(f"   🔍 Estimated vector search time: {metrics['vector_search_time']:.3f}s")
                    print(f"   🤖 Estimated generation time: {metrics['generation_time']:.3f}s")
                    print(f"   💰 RAG embedding cost: ${total_embedding_cost:.6f} (estimated)")
                    print(f"   💰 Estimated generation cost: ${metrics['generation_cost']:.6f}")

            else:
                # RAGSMALL_MATCH - không vào RAG_CHAT
                print("✅ Direct RAGSMALL match - No RAG_CHAT pipeline used")
                # Các metrics RAG_CHAT giữ nguyên giá trị 0

            # === PHASE 3: SIMILARITY CALCULATION ===
            print("📊 Phase 3: Answer Quality Assessment...")

            if metrics["actual_answer"] and expected_answer:
                cosine_sim = await self.calculate_cosine_similarity(metrics["actual_answer"], expected_answer)
                metrics["cosine_similarity"] = cosine_sim
                metrics["accuracy_score"] = 1 if cosine_sim >= 0.7 else 0

                print(f"   📈 Cosine similarity: {cosine_sim:.4f}")
                print(f"   ✅ Accuracy score: {metrics['accuracy_score']}")
            else:
                metrics["cosine_similarity"] = 0
                metrics["accuracy_score"] = 0
                metrics["errors"].append("Missing answer for similarity calculation")

        except Exception as e:
            logger.error(f"Error evaluating query {query_index + 1}: {e}")
            metrics["errors"].append(f"Evaluation error: {str(e)}")
            metrics["actual_answer"] = "ERROR"

        finally:
            # === FINAL METRICS ===
            total_end_time = time.time()
            metrics["total_time"] = total_end_time - total_start_time

            # Tính total cost (tránh trùng lặp embedding cost)
            if "FALLBACK" in metrics["route_used"] or "RAG_CHAT" in metrics["route_used"]:
                # RAG_CHAT flow: ragsmall + pre_retrieval + embedding + generation
                metrics["total_cost"] = (
                    metrics["ragsmall_search_cost"] +
                    metrics["pre_retrieval_cost"] +
                    metrics["embedding_cost"] +  # Đã tính từ subqueries
                    metrics["generation_cost"]
                )
            else:
                # RAGSMALL_MATCH flow: chỉ có ragsmall cost
                metrics["total_cost"] = metrics["ragsmall_search_cost"]

            print(f"\n📋 SUMMARY for Query {query_index + 1}:")
            print(f"   ⏱️  Total time: {metrics['total_time']:.3f}s")
            print(f"   💰 Total cost: ${metrics['total_cost']:.6f}")
            print(f"   🎯 Route used: {metrics['route_used']}")
            print(f"   ✅ Accuracy: {metrics['accuracy_score']}")

            if metrics["errors"]:
                print(f"   ⚠️  Errors: {len(metrics['errors'])}")

        return metrics

    async def run_evaluation(self, testcase_file: str = "testcase.xlsx"):
        """Chạy evaluation cho toàn bộ testcase"""
        print("🚀 Starting Detailed RAG Evaluation...")
        print(f"📁 Reading testcase from: {testcase_file}")

        # Đọc testcase
        try:
            df = pd.read_excel(testcase_file)
            print(f"📊 Loaded {len(df)} test cases")

            # Giả định có 2 cột: question và answer
            if 'question' not in df.columns or 'answer' not in df.columns:
                # Nếu không có tên cột chuẩn, lấy 2 cột đầu tiên
                df.columns = ['question', 'answer'] + list(df.columns[2:])
                print("⚠️  Using first 2 columns as question and answer")

        except Exception as e:
            logger.error(f"Error reading testcase file: {e}")
            return

        # Chạy evaluation cho từng query
        all_results = []
        total_queries = len(df)

        print(f"\n🔄 Processing {total_queries} queries...")

        for index, row in df.iterrows():
            question = str(row['question'])
            expected_answer = str(row['answer'])

            # Skip empty rows
            if not question or question.lower() in ['nan', 'none', '']:
                continue

            print(f"\n{'🔄' * 20} QUERY {index + 1}/{total_queries} {'🔄' * 20}")

            result = await self.evaluate_single_query(question, expected_answer, index)
            all_results.append(result)

            # Delay nhỏ để tránh rate limiting
            await asyncio.sleep(0.5)

        # === ANALYSIS & REPORTING ===
        print(f"\n{'🎯' * 20} ANALYSIS PHASE {'🎯' * 20}")
        await self.generate_report(all_results)

    async def generate_report(self, results: List[Dict[str, Any]]):
        """Tạo báo cáo chi tiết và summary"""
        print("📊 Generating detailed report...")

        # Tạo detailed DataFrame
        detailed_data = []
        for result in results:
            row = {
                "Query_Index": result["query_index"],
                "Question": result["question"],
                "Expected_Answer": result["expected_answer"],
                "Actual_Answer": result["actual_answer"],
                "Route_Used": result["route_used"],

                # Master chatbot phase
                "Master_Chatbot_Time_s": round(result["master_chatbot_time"], 4),
                "Ragsmall_Search_Cost_USD": round(result["ragsmall_search_cost"], 8),
                "Ragsmall_Similarity": round(result["ragsmall_similarity"], 4),

                # RAG_CHAT phases (0 if not used)
                "Pre_Retrieval_Time_s": round(result["pre_retrieval_time"], 4),
                "Embedding_Time_s": round(result["embedding_time"], 4),
                "Vector_Search_Time_s": round(result["vector_search_time"], 4),
                "Generation_Time_s": round(result["generation_time"], 4),

                # RAG_CHAT costs (0 if not used)
                "Pre_Retrieval_Cost_USD": round(result["pre_retrieval_cost"], 8),
                "Embedding_Cost_USD": round(result["embedding_cost"], 8),
                "Generation_Cost_USD": round(result["generation_cost"], 8),

                # Totals
                "Total_Time_s": round(result["total_time"], 4),
                "Total_Cost_USD": round(result["total_cost"], 8),

                # Quality metrics
                "Cosine_Similarity": round(result["cosine_similarity"], 4),
                "Accuracy_Score": result["accuracy_score"],
                "Chunks_Count": len(result["chunks_retrieved"]),
                "Has_Errors": len(result["errors"]) > 0,
                "Errors": "; ".join(result["errors"]) if result["errors"] else ""
            }
            detailed_data.append(row)

        detailed_df = pd.DataFrame(detailed_data)

        # Tính summary statistics
        summary_stats = {
            "Total_Queries": len(results),
            "Successful_Queries": len([r for r in results if not r["errors"]]),
            "Failed_Queries": len([r for r in results if r["errors"]]),

            # Route distribution
            "RAGSMALL_MATCH_Count": len([r for r in results if r["route_used"] == "RAGSMALL_MATCH"]),
            "RAG_CHAT_FALLBACK_Count": len([r for r in results if "FALLBACK" in r["route_used"]]),

            # Timing averages
            "Avg_Master_Chatbot_Time_s": np.mean([r["master_chatbot_time"] for r in results]),
            "Avg_Pre_Retrieval_Time_s": np.mean([r["pre_retrieval_time"] for r in results if r["pre_retrieval_time"] > 0]),
            "Avg_Embedding_Time_s": np.mean([r["embedding_time"] for r in results if r["embedding_time"] > 0]),
            "Avg_Vector_Search_Time_s": np.mean([r["vector_search_time"] for r in results if r["vector_search_time"] > 0]),
            "Avg_Generation_Time_s": np.mean([r["generation_time"] for r in results if r["generation_time"] > 0]),
            "Avg_Total_Time_s": np.mean([r["total_time"] for r in results]),

            # ✅ Thêm RAG Chat Latency và Ragsmall Latency
            "Avg_RAG_Chat_Latency_s": np.mean([
                r["pre_retrieval_time"] + r["embedding_time"] + r["vector_search_time"] + r["generation_time"]
                for r in results if "FALLBACK" in r["route_used"] or "RAG_CHAT" in r["route_used"]
            ]) if [r for r in results if "FALLBACK" in r["route_used"] or "RAG_CHAT" in r["route_used"]] else 0.0,

            "Avg_Ragsmall_Latency_s": np.mean([
                r["master_chatbot_time"] for r in results if r["route_used"] == "RAGSMALL_MATCH"
            ]) if [r for r in results if r["route_used"] == "RAGSMALL_MATCH"] else 0.0,

            # Cost averages
            "Avg_Ragsmall_Search_Cost_USD": np.mean([r["ragsmall_search_cost"] for r in results]),
            "Avg_Pre_Retrieval_Cost_USD": np.mean([r["pre_retrieval_cost"] for r in results if r["pre_retrieval_cost"] > 0]),
            "Avg_Embedding_Cost_USD": np.mean([r["embedding_cost"] for r in results if r["embedding_cost"] > 0]),
            "Avg_Generation_Cost_USD": np.mean([r["generation_cost"] for r in results if r["generation_cost"] > 0]),
            "Avg_Total_Cost_USD": np.mean([r["total_cost"] for r in results]),

            # ✅ Thêm RAG Chat Cost và Ragsmall Cost
            "Avg_RAG_Chat_Cost_USD": np.mean([
                r["total_cost"]  # ✅ Lấy total_cost (đã bao gồm tất cả) cho RAG_CHAT route
                for r in results if "FALLBACK" in r["route_used"] or "RAG_CHAT" in r["route_used"]
            ]) if [r for r in results if "FALLBACK" in r["route_used"] or "RAG_CHAT" in r["route_used"]] else 0.0,

            "Avg_Ragsmall_Cost_USD": np.mean([
                r["ragsmall_search_cost"]  # ✅ Chỉ lấy ragsmall cost cho RAGSMALL_MATCH route
                for r in results if r["route_used"] == "RAGSMALL_MATCH"
            ]) if [r for r in results if r["route_used"] == "RAGSMALL_MATCH"] else 0.0,

            # Totals
            "Total_Cost_USD": np.sum([r["total_cost"] for r in results]),
            "Total_Time_s": np.sum([r["total_time"] for r in results]),

            # Quality metrics
            "Avg_Ragsmall_Similarity": np.mean([r["ragsmall_similarity"] for r in results if r["ragsmall_similarity"] > 0]),
            "Avg_Cosine_Similarity": np.mean([r["cosine_similarity"] for r in results]),
            "Accuracy_Rate": np.mean([r["accuracy_score"] for r in results]),
            "High_Accuracy_Count": len([r for r in results if r["accuracy_score"] == 1]),
        }

        # Handle NaN values for cases where no RAG_CHAT was used
        for key in summary_stats:
            if np.isnan(summary_stats[key]):
                summary_stats[key] = 0.0

        # Tạo summary DataFrame
        summary_df = pd.DataFrame([summary_stats])

        # === SAVE TO EXCEL ===
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rag_evaluation_detailed_{timestamp}.xlsx"

        print(f"💾 Saving report to: {filename}")

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: Detailed results
            detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)

            # Sheet 2: Summary statistics
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)

            # Sheet 3: Chunks information (từ retrieved_docs thực tế)
            chunks_data = []
            for i, result in enumerate(results):  # Tất cả queries, không chỉ 10 đầu
                for chunk in result["chunks_retrieved"]:
                    chunks_data.append({
                        "Query_Index": result["query_index"],
                        "Chunk_Index": chunk.get("chunk_index", 0),
                        "Content": chunk.get("content", ""),
                        "Content_Length": chunk.get("content_length", 0),
                        "Metadata": str(chunk.get("metadata", {}))
                    })

            if chunks_data:
                chunks_df = pd.DataFrame(chunks_data)
                chunks_df.to_excel(writer, sheet_name='Retrieved_Chunks', index=False)
            else:
                # Tạo empty sheet nếu không có chunks
                empty_chunks_df = pd.DataFrame({
                    "Query_Index": [],
                    "Chunk_Index": [],
                    "Content": [],
                    "Content_Length": [],
                    "Metadata": []
                })
                empty_chunks_df.to_excel(writer, sheet_name='Retrieved_Chunks', index=False)

        # === PRINT SUMMARY ===
        print(f"\n{'🎯' * 20} EVALUATION SUMMARY {'🎯' * 20}")
        print(f"📊 Total Queries: {summary_stats['Total_Queries']}")
        print(f"✅ Successful: {summary_stats['Successful_Queries']}")
        print(f"❌ Failed: {summary_stats['Failed_Queries']}")
        print(f"🏠 RAGSMALL matches: {summary_stats['RAGSMALL_MATCH_Count']}")
        print(f"🔄 RAG_CHAT fallbacks: {summary_stats['RAG_CHAT_FALLBACK_Count']}")
        print(f"🎯 Accuracy Rate: {summary_stats['Accuracy_Rate']:.2%}")
        print(f"📈 Avg Cosine Similarity: {summary_stats['Avg_Cosine_Similarity']:.4f}")
        print(f"⏱️  Avg Total Time: {summary_stats['Avg_Total_Time_s']:.3f}s")
        print(f"💰 Avg Total Cost: ${summary_stats['Avg_Total_Cost_USD']:.6f}")
        print(f"💰 Total Cost: ${summary_stats['Total_Cost_USD']:.6f}")
        print(f"📁 Report saved: {filename}")
        print(f"{'🎯' * 60}")

async def main():
    """Main function to run the evaluation"""
    evaluation = DetailedEvaluation()
    await evaluation.run_evaluation()

if __name__ == "__main__":
    asyncio.run(main())
