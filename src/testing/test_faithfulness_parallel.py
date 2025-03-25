import json
# from dotenv import load_dotenv
from multiprocessing import Pool
from pipeline.retriever import Retriever
from pipeline.generator import Generator
from evaluation.faithfulness_evaluation import FaithfulnessEvaluator
from log_manager import setup_logger
from evaluation.evaluation_model import LMStudioEvaluationModel
from evaluation.evaluation_logger import EvaluationLogger

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI")

# Main logger
logger = setup_logger("logs/test_faithfulness_parallel.log")
parallel_logger = EvaluationLogger(eval_type="faithfulness")

def evaluate_entry(entry):
    try:
        retriever = Retriever()
        generator = Generator()
        evaluation_model = LMStudioEvaluationModel("http://localhost:1234/v1/chat/completions")
        faithfulness_eval = FaithfulnessEvaluator(retriever, generator, evaluation_model, embedding_model="BAAI/bge-base-en")

        query = entry["question"]
        ground_truth_answer = entry["answer"]
        retrieved_chunks = retriever.retrieve(query, top_k=5)
        generated_answer = generator.generate_answer(query, retrieved_docs=retrieved_chunks)

        # Non-LLM metrics
        answer_similarity = faithfulness_eval.answer_chunk_similarity(retrieved_chunks, generated_answer)
        faithful_coverage = faithfulness_eval.compute_faithful_coverage(ground_truth_answer, generated_answer)

        # LLM-based metrics
        faithfulness_score_llm = faithfulness_eval.llm_as_judge(retrieved_chunks, generated_answer)
        faithful_coverage_llm = faithfulness_eval.llm_faithful_coverage(query, ground_truth_answer, generated_answer)
    

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

        parallel_logger.log(result_data)
        return result_data

    except Exception as e:
        print(f"❌ Fatal error in entry: {entry['question']} → {e}")
        parallel_logger.log_error(entry["question"], str(e))
        return None


if __name__ == "__main__":
    with open("data/bleach_ground_truth_qna.json", "r") as f:
        ground_truth_qna = json.load(f)

    print("🚀 Starting parallel faithfulness evaluation with 4 workers...")
    with Pool(processes=4) as pool:
        pool.map(evaluate_entry, ground_truth_qna)

    print("✅ All evaluations complete. Saving to Excel...")
    parallel_logger.log_to_excel()
    print("📊 Excel saved. Logs written to logs/test_faithfulness_parallel.log")