import os
import json
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from pipeline.retrieval import Retriever
from pipeline.llm import LLM
from evaluation.evaluation_logger import EvaluationLogger


# Load environment variables
load_dotenv()

# Set up Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ No API key found! Set GOOGLE_API_KEY environment variable.")
genai.configure(api_key=api_key)


import chromadb
from pipeline.retrieval import Retriever

# ✅ Correctly initialize ChromaDB and pass the collection
db_dir = "bleach_wiki/vector_db"  # Ensure this path is correct
client = chromadb.PersistentClient(path=db_dir)

# ✅ Retrieve the actual collection object instead of passing a string
collection = client.get_collection(name="bleach_wiki_embeddings")  

retriever = Retriever(vector_store=collection)  # ✅ Pass the ChromaDB collection, NOT a string

# Initialize components
retriever = Retriever(vector_store="bleach_wiki_embeddings")  
llm = LLM()

def get_text_embedding(text):
    """
    Generates an embedding for a given text using Gemini.

    Args:
        text (str): Input text.

    Returns:
        List[float]: Embedding vector.
    """
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return response["embedding"]
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return None


def compute_cosine_similarity(query_embedding, chunk_embeddings):
    """
    Computes cosine similarity between query embedding and retrieved chunk embeddings.

    Args:
        query_embedding (numpy.ndarray): The embedding of the LLM response.
        chunk_embeddings (numpy.ndarray): The embeddings of retrieved chunks.

    Returns:
        float: Cosine similarity score.
    """
    query_embedding = np.array(query_embedding).reshape(1, -1)
    chunk_embeddings = np.array(chunk_embeddings)
    return float(cosine_similarity(query_embedding, chunk_embeddings).mean())


def evaluate_faithfulness(query):
    """
    Runs the faithfulness evaluation pipeline.

    Args:
        query (str): The user query.
    """
    # Step 1: Retrieve Relevant Documents
    retrieved_docs = retriever.retrieve_documents(query)
    if not retrieved_docs["documents"]:
        print("❌ No relevant documents found.")
        return
    
    print("\n🔍 **Top Retrieved Chunks:**")
    for i, doc in enumerate(retrieved_docs["metadatas"], start=1):
        print(f"{i}. {doc['title']} -> {doc['text'][:200]}...")

    # Step 2: Generate Answer using LLM
    context = "\n".join([doc["text"] for doc in retrieved_docs["metadatas"]])
    generated_answer = llm.generate_response(query, context)

    print("\n🤖 **Generated Answer:**")
    print(generated_answer)

    # Step 3: Compute Faithfulness Score
    retrieved_texts = [doc["text"] for doc in retrieved_docs["metadatas"]]
    retrieved_texts.append(generated_answer)  # Add generated answer for comparison
    embeddings = [get_text_embedding(text) for text in retrieved_texts]

    if None in embeddings:
        print("❌ Failed to generate embeddings for some texts.")
        return

    answer_embedding = np.array(embeddings[-1])  # Last embedding is the generated answer
    retrieved_embeddings = np.array(embeddings[:-1])  # Exclude last one

    faithfulness_score = compute_cosine_similarity(answer_embedding, retrieved_embeddings)
    print(f"\n📊 **Faithfulness Score:** {faithfulness_score:.2f}")

    # Save evaluation results
    save_evaluation_results(query, retrieved_docs, generated_answer, faithfulness_score)


def save_evaluation_results(query, retrieved_docs, answer, similarity_score):
    """
    Saves the evaluation results to a text file.

    Args:
        query (str): User query.
        retrieved_docs (List[Dict]): Retrieved documents.
        answer (str): Generated answer.
        similarity_score (float): Cosine similarity score.
    """
    with open("e_results.txt", "a", encoding="utf-8") as f:
        f.write(f"\nQuery: {query}\n")
        f.write("Retrieved Chunks:\n")
        for i, doc in enumerate(retrieved_docs["metadatas"], start=1):
            f.write(f"{i}. {doc['title']} -> {doc['text'][:200]}...\n")
        f.write(f"Generated Answer: {answer}\n")
        f.write(f"Faithfulness Score: {similarity_score:.2f}\n")
        f.write("-" * 50 + "\n")
    print("✅ Results saved to e_results.txt")


if __name__ == "__main__":
    query = input("🔍 Enter your question: ")
    evaluate_faithfulness(query)
