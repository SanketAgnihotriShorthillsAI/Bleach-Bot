import json
from pipeline.retriever import Retriever  # Adjusted import as per your retrieval.py
from pipeline.generator import Generator  # Using your custom LLM class
from evaluation.faithfulness_evaluation import FaithfulnessEvaluator  # Adjusted for your refactored module

# ✅ Initialize retriever, generator, and evaluator
retriever = Retriever()
generator = Generator()
faithfulness_eval = FaithfulnessEvaluator(retriever, generator)

# ✅ Load Ground Truth QnA from Bleach Wiki
with open("data/bleach_ground_truth_qna.json", "r") as f:
    ground_truth_qna = json.load(f)

# ✅ Process each query
for i, qna in enumerate(ground_truth_qna):
    query = qna["question"]
    ground_truth_answer = qna["answer"]

    print(f"\n🔍 Running Faithfulness Evaluation for Query {i + 1}/{len(ground_truth_qna)}: {query}")

    # ✅ Evaluate faithfulness using the tailored evaluator
    result = faithfulness_eval.evaluate_faithfulness(query, ground_truth_answer, top_k=5)

    # ✅ Log results if evaluation was successful
    if result:
        faithfulness_eval.logger.log(result)

print("\n✅ Faithfulness Evaluation Complete! Results are logged.")
