import json
from pipeline.retriever import Retriever  # Adjusted import as per your retrieval.py
from pipeline.generator import Generator  # Using your custom LLM class
from evaluation.faithfulness_evaluation import FaithfulnessEvaluator  # Adjusted for your refactored module
from log_manager import setup_logger
from evaluation.evaluation_model import LMStudioEvaluationModel

logger = setup_logger("logs/test_faithfulness.log")

# ✅ Initialize retriever, generator, and evaluator
retriever = Retriever()
generator = Generator()
evaluation_model = LMStudioEvaluationModel("http://localhost:1234/v1/chat/completions")
faithfulness_eval = FaithfulnessEvaluator(retriever, generator, evaluation_model)

# ✅ Load Ground Truth QnA from Bleach Wiki
with open("data/bleach_ground_truth_qna.json", "r") as f:
    ground_truth_qna = json.load(f)

# ✅ Process each query
for i, qna in enumerate(ground_truth_qna):
    query = qna["question"]
    ground_truth_answer = qna["answer"]

    # Retrieve Chunks ONCE and pass them to all methods
    retrieved_chunks = retriever.retrieve(query, top_k=5)

    # Generate the answer for the given query
    generated_answer = generator.generate_answer(query, retrieved_chunks)

    # Compute faithfulness evaluation metrics (Non-LLM)
    answer_similarity = faithfulness_eval.answer_chunk_similarity(retrieved_chunks, generated_answer)
    faithful_coverage = faithfulness_eval.compute_faithful_coverage(ground_truth_answer, generated_answer)
    # negative_faithfulness = faithfulness_eval.compute_negative_faithfulness(query, retrieved_chunks, generated_answer)

    # Compute LLM-based faithfulness evaluation metrics, with error handling for API exhaustion
    faithfulness_score_llm = faithfulness_eval.llm_as_judge(retrieved_chunks, generated_answer)
    faithful_coverage_llm = faithfulness_eval.llm_faithful_coverage(query, ground_truth_answer, generated_answer)
    
    # Prepare the result data for logging
    result_data = {
        "query": query,
        "ground_truth_answer": ground_truth_answer,
        "generated_answer": faithfulness_eval.generator.generate_answer(query, retrieved_chunks),
        "answer_chunk_similarity": answer_similarity,
        "faithful_coverage": faithful_coverage,
        # "negative_faithfulness": negative_faithfulness,
        "faithfulness_score_llm": faithfulness_score_llm,
        "faithful_coverage_llm": faithful_coverage_llm
    }

    # Log the results
    faithfulness_eval.logger.log(result_data)

logger.info("Faithfulness Evaluation completed!")
faithfulness_eval.logger.log_to_excel()
logger.info("Results saved to excel.")