from langchain_community.vectorstores import Chroma
from typing import Dict, Any
import chromadb  

class Retriever:
    """
    Handles retrieval of relevant documents from a ChromaDB vector store.
    """

    def __init__(self, vector_store) -> None:
        """
        Initializes the retriever with a vector store.

        Args:
            vector_store: An instance of the Chroma collection.
        """
        if not isinstance(vector_store, Chroma):
            raise TypeError(f"❌ Expected a ChromaDB collection, but got {type(vector_store)} instead.")

        self.vector_store = vector_store
        self.retriever = self.vector_store.as_retriever()

    def retrieve_documents(self, query: str) -> Dict[str, Any]:
        """
        Retrieves the most relevant documents from the vector store.

        Args:
            query (str): The search query.

        Returns:
            Dict[str, Any]: Retrieved documents and metadata.
        """
        print(f"🔍 Retrieving documents for query: {query}")

        results = self.retriever.get_relevant_documents(query)

        if not results:
            print("⚠️ No relevant documents found.")
            return {"documents": [], "metadatas": []}

        # Format the retrieved data
        docs = [
            {
                "title": doc.metadata.get("title", "Unknown"),
                "source": doc.metadata.get("source", "Unknown"),
                "text": doc.page_content,
            }
            for doc in results
        ]
        
        return {"documents": results, "metadatas": docs}
