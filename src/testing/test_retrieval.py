import json
import os
from pipeline.retriever import Retriever
from pipeline.generator import Generator
from evaluation.retrieval_eval import RetrievalEvaluator
from evaluation.evaluation_logger import EvaluationLogger
from evaluation.evaluation_model import LMStudioEvaluationModel
from log_manager import setup_logger

logger = setup_logger("logs/test_retrieval.log")
logger = EvaluationLogger(eval_type="retrieval")

# Initialize retriever, generator, and evaluator
retriever = Retriever()
generator = Generator()
evaluation_model = LMStudioEvaluationModel("http://127.0.0.1:1234/v1/chat/completions")
retrieval_eval = RetrievalEvaluator(retriever, generator,evaluation_model)

# Load Ground Truth QnA
with open("data/bleach_ground_truth_qna.json", "r") as f:
    ground_truth_qna = json.load(f)

# Loop through all QnA pairs
for idx, qna in enumerate(ground_truth_qna):
    query = qna["question"]
    ground_truth_answer = qna["answer"]

    # ✅ Retrieve Chunks ONCE and pass them to all methods
    retrieved_chunks = retriever.retrieve(query, top_k=5)

    print(f"\n🔎 Evaluating QnA Pair {idx + 1}/{len(ground_truth_qna)}")
    retrieved_chunks = retriever.retrieve(query, top_k=5)

    # ✅ Compute retrieval evaluation metrics (Cosine-Based)
    context_precision = retrieval_eval.compute_context_precision(query, retrieved_chunks)
    context_recall = retrieval_eval.compute_context_recall(query, ground_truth_answer, retrieved_chunks)
    # context_overlap = retrieval_eval.compute_context_overlap(query, ground_truth_answer, retrieved_chunks)
    # negative_retrieval = retrieval_eval.compute_negative_retrieval(query, retrieved_chunks)

    # ✅ Compute LLM-based retrieval evaluation metrics
    context_precision_llm = retrieval_eval.compute_context_precision_with_llm(query, retrieved_chunks)
    context_recall_llm = retrieval_eval.compute_context_recall_with_llm(query, ground_truth_answer, retrieved_chunks)
    retrieval_precision_llm = retrieval_eval.compute_retrieval_precision_with_llm(query, retrieved_chunks)
    # context_overlap_llm = retrieval_eval.compute_context_overlap_with_llm(query, ground_truth_answer, retrieved_chunks)
    negative_retrieval_llm = retrieval_eval.compute_negative_retrieval_with_llm(query, retrieved_chunks)

    # ✅ Print results
    print("\n📊 Final Retrieval Evaluation Scores (Cosine-Based):")
    print(f"✅ Context Precision: {context_precision['combined_precision_score']:.2f}")
    print(f"✅ Context Recall: {context_recall:.2f}")

    # print(f"✅ Context Overlap Score (ROUGE-L): {context_overlap:.2f}")
    # print(f"✅ Negative Retrieval Score: {negative_retrieval:.2f}")

    print("\n🤖 Final Retrieval Evaluation Scores (LLM-Based):")
    print(f"🤖 Context Precision (LLM): {context_precision_llm}")
    print(f"🤖 Context Recall (LLM): {context_recall_llm}")
    print(f"🤖 Retrieval Precision (LLM): {retrieval_precision_llm}")
    # print(f"🤖 Context Overlap Score (LLM): {context_overlap_llm:.2f}")
    print(f"🤖 Negative Retrieval Score (LLM): {negative_retrieval_llm}")

    # ✅ Log results for this query
    result = {
        "query": query,
        "Ground Truth Answer": ground_truth_answer,
        "Context Precision":context_precision["combined_precision_score"], # Individual cosine, BM25 scores can also be extracted
        "Context Recall": context_recall,
        # "context_overlap": context_overlap,
        # "negative_retrieval": negative_retrieval,
        "Context Precision (llm)": context_precision_llm,
        "Context Recall (llm)": context_recall_llm,
        "Retrieval Precision (llm)": retrieval_precision_llm,
        # "context_overlap_llm": context_overlap_llm,
        "Negative Retrieval (llm)": negative_retrieval_llm
    }
    logger.log(result)

# ✅ Convert all logs to Excel (Run separately if needed)
logger.log_to_excel()