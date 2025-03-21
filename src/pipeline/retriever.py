import os
import chromadb
import google.generativeai as genai
from typing import List, Dict, Any
from dotenv import load_dotenv
# import logging

# # Suppress ChromaDB warnings
# logging.getLogger("chromadb.segment.impl.vector.local_persistent_hnsw").setLevel(logging.ERROR)

class Retriever:
    def __init__(self, db_dir: str = "../bleach_wiki/vector_db"):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("No API key found! Set GOOGLE_API_KEY environment variable.")

        genai.configure(api_key=api_key)

        # Set up ChromaDB client
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.client.get_collection(name="bleach_wiki_embeddings")

    def get_query_embedding(self, query: str) -> List[float]:
        response = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        return response["embedding"]

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.get_query_embedding(query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

        retrieved_docs = []
        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
            retrieved_docs.append({"text": doc, "title": metadata.get("title", "Unknown")})

        return retrieved_docs
