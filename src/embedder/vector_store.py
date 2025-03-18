import os
import json
import chromadb
from tqdm import tqdm
from typing import List, Dict, Any, Optional


class BleachWikiVectorStore:
    """
    A class for managing a persistent vector database using ChromaDB.

    This class loads precomputed embeddings from JSONL files and stores them 
    in a ChromaDB vector database for efficient retrieval.

    Attributes:
        input_dir (str): Directory containing JSONL embedding files.
        db_dir (str): Directory where the ChromaDB vector database is stored.
        client (chromadb.PersistentClient): ChromaDB persistent client instance.
        collection (chromadb.Collection): Collection used to store and retrieve embeddings.
    """

    def __init__(self, input_dir: str = "bleach_wiki/embeddings", db_dir: str = "bleach_wiki/vector_db") -> None:
        """
        Initializes the vector store for Bleach Wiki.

        Args:
            input_dir (str): Directory containing JSONL embedding files.
            db_dir (str): Directory to store the ChromaDB vector database.
        """
        self.input_dir = input_dir
        self.db_dir = db_dir

        # Ensure the database directory exists
        os.makedirs(self.db_dir, exist_ok=True)

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.client.get_or_create_collection(name="bleach_wiki_embeddings")

    def as_retriever(self) -> Any:
        """
        Converts the vector store into a retriever for querying.

        Returns:
            Any: A retriever object for querying the stored embeddings.
        """
        return self.collection.as_retriever()

    def load_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Loads JSONL data from a file and returns a list of embeddings.

        Args:
            filepath (str): Path to the JSONL file.

        Returns:
            List[Dict[str, Any]]: A list of embedding data dictionaries.
        """
        data = []
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                for line in file:
                    data.append(json.loads(line.strip()))
            return data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return []

    def add_to_vector_db(self, embeddings: List[Dict[str, Any]]) -> None:
        """
        Adds embeddings to the ChromaDB vector database.

        Args:
            embeddings (List[Dict[str, Any]]): A list of embedding dictionaries.
        """
        for entry in tqdm(embeddings, desc="Storing embeddings in ChromaDB"):
            chunk_id = entry.get("chunk_id", "unknown")
            vector = entry.get("embedding", [])
            document = entry.get("text", "")
            metadata = {
                "source": entry.get("source", "Bleach Fandom Wiki"),
                "title": entry.get("title", "Unknown"),
                "text": document
            }

            if vector:  # Ensure the embedding is not empty
                self.collection.add(
                    ids=[chunk_id],
                    embeddings=[vector],
                    documents=[document],
                    metadatas=[metadata]
                )

    def process_files(self) -> None:
        """
        Processes all JSONL embedding files and stores them in ChromaDB.
        """
        files = [f for f in os.listdir(self.input_dir) if f.endswith(".jsonl")]
        for file in files:
            filepath = os.path.join(self.input_dir, file)
            print(f"Processing {file} for vector storage...")

            embeddings = self.load_jsonl(filepath)
            if embeddings:
                self.add_to_vector_db(embeddings)
                print(f"Successfully stored {len(embeddings)} embeddings from {file}")

    def run(self) -> None:
        """
        Executes the entire process of loading embeddings and storing them in ChromaDB.
        """
        self.process_files()
        print("All Bleach Wiki embeddings stored successfully!")


# Run the vector store
if __name__ == "__main__":
    vector_store = BleachWikiVectorStore()
    vector_store.run()
