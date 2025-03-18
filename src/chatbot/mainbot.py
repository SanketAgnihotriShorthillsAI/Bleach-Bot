import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables (for API key)
load_dotenv()

# Set up Gemini API client
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("No API key found! Set GOOGLE_API_KEY environment variable.")

genai.configure(api_key=api_key)

# Set up ChromaDB client
db_dir = "bleach_wiki/vector_db"  # Ensure this is the correct path
client = chromadb.PersistentClient(path=db_dir)
collection = client.get_collection(name="bleach_wiki_embeddings")


def get_query_embedding(query: str) -> List[float]:
    """
    Generates an embedding for the given query using Gemini.

    Args:
        query (str): The user query.

    Returns:
        List[float]: The generated query embedding.
    """
    response = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    return response["embedding"]


def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieves the most relevant documents from the ChromaDB vector store.

    Args:
        query (str): The search query.
        top_k (int, optional): The number of top results to return. Defaults to 5.

    Returns:
        List[Dict[str, Any]]: A list of retrieved documents with metadata.
    """
    query_embedding = get_query_embedding(query)
    
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    retrieved_docs = []
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        retrieved_docs.append({"text": doc, "title": metadata.get("title", "Unknown")})

    return retrieved_docs


def generate_answer(query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    """
    Generates an answer using Gemini Pro based on the retrieved context.

    Args:
        query (str): The user query.
        retrieved_docs (List[Dict[str, Any]]): The retrieved context documents.

    Returns:
        str: The generated answer from Gemini Pro.
    """
    if not retrieved_docs:
        return "I don't know. No relevant information found."

    context = "\n".join([doc["text"] for doc in retrieved_docs])

    prompt = (
        "You are a knowledgeable assistant, trained on Bleach Wiki. "
        "Go through all the provided context and answer the query. "
        "If the context doesn't contain enough information, say \"I don't know\" "
        "instead of making up an answer by yourself.\n\n"
        f"Context:\n{context}\n\nQuery: {query}\n"
    )

    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(prompt)

    return response.text


if __name__ == "__main__":
    query = input("Ask me a question: ")
    retrieved_docs = retrieve(query)

    print("\nTop Retrieved Chunks:")
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"{i}. {doc['title']} -> {doc['text'][:200]}...")

    answer = generate_answer(query, retrieved_docs)

    print("\nGenerated Answer:")
    print(answer)