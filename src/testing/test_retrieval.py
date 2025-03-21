import json
from pipeline.retriever import Retriever
from pipeline.generator import Generator
from evaluation.retrieval_eval import RetrievalEvaluator

# Initialize retriever, generator, and evaluator
retriever = Retriever()
generator = Generator()
retrieval_eval = RetrievalEvaluator(retriever, generator)

# Load Ground Truth QnA
with open("data/bleach_ground_truth_qna.json", "r") as f:
    ground_truth_qna = json.load(f)

# Run evaluation for a sample query
query = ground_truth_qna[0]["question"]
ground_truth_answer = ground_truth_qna[0]["answer"]

# ✅ Retrieve Chunks ONCE and pass them to all methods
retrieved_chunks = retriever.retrieve(query, top_k=5)

# Loop through all QnA pairs
for idx, qna in enumerate(ground_truth_qna):
    query = qna["question"]
    ground_truth_answer = qna["answer"]

    print(f"\n🔎 Evaluating QnA Pair {idx + 1}/{len(ground_truth_qna)}")
    retrieved_chunks = retriever.retrieve(query, top_k=5)

    # ✅ Compute retrieval evaluation metrics (Cosine-Based)
    context_recall = retrieval_eval.compute_context_recall(query, ground_truth_answer, retrieved_chunks)
    context_precision = retrieval_eval.compute_context_precision(query, retrieved_chunks)
    context_overlap = retrieval_eval.compute_context_overlap(query, ground_truth_answer, retrieved_chunks)
    negative_retrieval = retrieval_eval.compute_negative_retrieval(query, retrieved_chunks)

    # ✅ Compute LLM-based retrieval evaluation metrics
    context_precision_llm = retrieval_eval.compute_context_precision_with_llm(query, retrieved_chunks)
    context_recall_llm = retrieval_eval.compute_context_recall_with_llm(query, ground_truth_answer, retrieved_chunks)
    retrieval_precision_llm = retrieval_eval.compute_retrieval_precision_with_llm(query, retrieved_chunks)
    context_overlap_llm = retrieval_eval.compute_context_overlap_with_llm(query, ground_truth_answer, retrieved_chunks)
    negative_retrieval_llm = retrieval_eval.compute_negative_retrieval_with_llm(query, retrieved_chunks)

    # ✅ Print results
    print("\n📊 Final Retrieval Evaluation Scores (Cosine-Based):")
    print(f"✅ Context Precision: {context_precision:.2f}")
    print(f"✅ Context Recall: {context_recall:.2f}")

    print(f"✅ Context Overlap Score (ROUGE-L): {context_overlap:.2f}")
    print(f"✅ Negative Retrieval Score: {negative_retrieval:.2f}")

    print("\n🤖 Final Retrieval Evaluation Scores (LLM-Based):")
    print(f"🤖 Context Precision (LLM): {context_precision_llm:.2f}")
    print(f"🤖 Context Recall (LLM): {context_recall_llm:.2f}")
    print(f"🤖 Retrieval Precision (LLM): {retrieval_precision_llm:.2f}")
    print(f"🤖 Context Overlap Score (LLM): {context_overlap_llm:.2f}")
    print(f"🤖 Negative Retrieval Score (LLM): {negative_retrieval_llm:.2f}")

