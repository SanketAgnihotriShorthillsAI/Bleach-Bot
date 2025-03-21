import json
import os
from dotenv import load_dotenv
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer  # ROUGE-L for faithfulness evaluation
from pipeline.retriever import Retriever
from pipeline.generator import Generator
from evaluation.evaluation_logger import EvaluationLogger
load_dotenv()

class FaithfulnessEvaluator:
    """
    Evaluates the faithfulness of LLM-generated responses for the Bleach Wiki RAG bot.
    """

    def __init__(self, retriever: Retriever, generator: Generator, embedding_model="BAAI/bge-base-en"):
        self.retriever = retriever
        self.generator = generator
        self.logger = EvaluationLogger(eval_type="faithfulness")
        self.model = SentenceTransformer(embedding_model)  # Embedding model for semantic similarity
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)  # ROUGE-L Scorer

    def evaluate_faithfulness(self, query, ground_truth_answer, top_k=5):
        """
        Evaluates the faithfulness of the generated answer against the retrieved context.
        """
        print(f"\n🔍 Evaluating Faithfulness for Query: {query}")

        # Retrieve relevant chunks from the Bleach Wiki
        retrieved_chunks = self.retriever.retrieve(query, top_k=top_k)
        if not retrieved_chunks:
            print("⚠️ No retrieved chunks found.")
            return None

        # Generate response using the LLM
        generated_answer = self.generator.generate_answer(query, retrieved_chunks)

        # Compute faithfulness metrics
        answer_chunk_similarity = self.answer_chunk_similarity("\n".join([doc["text"] for doc in retrieved_chunks]), generated_answer)
        faithful_coverage = self.compute_faithful_coverage(ground_truth_answer, generated_answer)
        negative_faithfulness = self.compute_negative_faithfulness(query, generated_answer)

        # LLM-based faithfulness evaluation
        faithfulness_llm = self.llm_as_judge(retrieved_chunks, generated_answer)
        faithful_coverage_llm = self.llm_faithful_coverage(ground_truth_answer, generated_answer)

        print("\n📊 Faithfulness Evaluation Results:")
        print(f"✅ Answer-Chunk Similarity: {answer_chunk_similarity:.2f}")
        print(f"✅ Faithful Coverage (ROUGE-L): {faithful_coverage:.2f}")
        print(f"✅ Negative Faithfulness: {negative_faithfulness:.2f}")
        print(f"🤖 Faithfulness (LLM): {faithfulness_llm}")
        print(f"🤖 Faithful Coverage (LLM): {faithful_coverage_llm}")

        return {
            "query": query,
            "generated_answer": generated_answer,
            "ground_truth_answer": ground_truth_answer,
            "answer_chunk_similarity": float(answer_chunk_similarity),
            "faithful_coverage": float(faithful_coverage),
            "negative_faithfulness": float(negative_faithfulness),
            "faithfulness_llm": faithfulness_llm,
            "faithful_coverage_llm": faithful_coverage_llm
        }

    # Non-LLM Based Methods

    def answer_chunk_similarity(self, retrieved_chunks, generated_answer):
        """
        Measures the cosine similarity between generated answer and retrieved chunks.
        """
        answer_embedding = self.model.encode([generated_answer], normalize_embeddings=True)
        chunk_embedding = self.model.encode([" ".join(retrieved_chunks)], normalize_embeddings=True)
        return float(cosine_similarity(answer_embedding, chunk_embedding)[0][0]) * 10

    def compute_faithful_coverage(self, ground_truth_answer, generated_answer):
        """
        Computes the ROUGE-L score between the generated answer and ground truth.
        """
        rouge_scores = self.rouge_scorer.score(ground_truth_answer, generated_answer)
        return rouge_scores["rougeL"].fmeasure * 10

    def compute_negative_faithfulness(self, query, generated_answer):
        """
        Checks if the generated answer contains information not related to the query.
        """
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        answer_embedding = self.model.encode([generated_answer], normalize_embeddings=True)
        return (1 - cosine_similarity(query_embedding, answer_embedding)[0][0]) * 10

    # LLM-Based Methods

    def llm_as_judge(self, retrieved_chunks, generated_answer):
        """
        Uses LLM to evaluate the faithfulness of the generated response.
        """
        prompt = f"""
        Evaluate the following generated answer's faithfulness to the retrieved context.

        Retrieved Context:
        {retrieved_chunks}

        Generated Answer:
        {generated_answer}

        Provide a score from 0 to 10:
        - 10: Perfectly faithful
        - 5: Somewhat faithful with extra information
        - 0: Completely unfaithful
        """
        try:
            response = self.generator.generate_answer(prompt, retrieved_chunks)
            return self._parse_llm_score(response)
        except Exception as e:
            print(f"⚠️ Error in LLM Call: {e} - Key exhausted.")
            return "Key exhausted"

    def llm_faithful_coverage(self, ground_truth_answer, generated_answer):
        """
        Uses LLM to evaluate the faithful coverage of the generated response.
        """
        prompt = f"""
        Evaluate the overlap of the following generated answer with the ground truth answer.

        Ground Truth Answer:
        {ground_truth_answer}

        Generated Answer:
        {generated_answer}

        ONLY provide a numeric score from 0 to 10. No additional explanation."
        """
        try:
            response = self.generator.model.generate_content(prompt)
            print(f"🔹 LLM Response for Faithful Coverage: '{response.text}'")
            return self._parse_llm_score(response.text)
        except Exception as e:
            print(f"⚠️ Error in LLM Call: {e} - Key exhausted.")
            return "Key exhausted"

    # Helper function
    def _parse_llm_score(self, response):
        """
        Parses the LLM's response to extract a numeric score.
        """
        response = response.strip()  # Remove leading/trailing whitespaces

        # Attempt direct float conversion
        try:
            return float(response)
        except ValueError:
            pass  # Continue to regex extraction

        # Extract the first valid float or integer
        match = re.search(r"(^\d+(\.\d+)?)|(\d+(\.\d+)?)", response)  # First or any valid number
        if match:
            return float(match.group(0)) 

        print(f"⚠️ Unable to parse a valid score from LLM response: '{response}'")
        return -1  # Return -1 if no valid number is found


