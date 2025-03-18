import os
import json
import google.generativeai as genai
import jsonlines
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Dict, Optional

# Load environment variables (for API key)
load_dotenv()


class EmbeddingGenerator:
    """
    A class for generating text embeddings using Google's Gemini model.

    This class reads chunked JSON files, generates embeddings using Gemini,
    and stores the embeddings in JSONL format for efficient retrieval.

    Attributes:
        input_dir (str): Directory containing chunked JSON files.
        output_dir (str): Directory to store the embedding JSONL files.
        model (str): Gemini embedding model name.
    """

    def __init__(
        self,
        input_dir: str = "bleach_wiki/chunks",
        output_dir: str = "bleach_wiki/embeddings",
        model: str = "models/embedding-001",
    ) -> None:
        """
        Initializes the EmbeddingGenerator with directories and embedding model.

        Args:
            input_dir (str): Path to directory containing chunked JSON files.
            output_dir (str): Path to directory for saving embeddings.
            model (str): Name of the Gemini embedding model.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model = model

        os.makedirs(self.output_dir, exist_ok=True)

        # Load API key for Google Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("No API key found! Set GOOGLE_API_KEY environment variable.")

        genai.configure(api_key=api_key)

    def load_json(self, filepath: str) -> Optional[List[Dict]]:
        """Loads a JSON file and returns its contents."""
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def save_jsonl(self, filename: str, data: List[Dict]) -> None:
        """Saves the list of embeddings to a JSONL file."""
        output_path = os.path.join(self.output_dir, filename.replace(".json", ".jsonl"))
        try:
            with jsonlines.open(output_path, "w") as writer:
                writer.write_all(data)
            print(f"Embeddings saved to {output_path}")
        except Exception as e:
            print(f"Error saving embeddings for {filename}: {e}")

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generates an embedding vector for the given text using Gemini."""
        try:
            response = genai.embed_content(model=self.model, content=text, task_type="retrieval_document")
            return response["embedding"]
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def is_already_embedded(self, filename: str) -> bool:
        """
        Checks if the file is already embedded by comparing timestamps.

        Args:
            filename (str): The name of the JSON file to check.

        Returns:
            bool: True if already embedded, False otherwise.
        """
        raw_file_path = os.path.join(self.input_dir, filename)
        embedded_file_path = os.path.join(self.output_dir, filename.replace(".json", ".jsonl"))

        # Check if embedded file exists and compare modification times
        if os.path.exists(embedded_file_path):
            raw_mtime = os.path.getmtime(raw_file_path)
            embedded_mtime = os.path.getmtime(embedded_file_path)
            if embedded_mtime >= raw_mtime:
                return True  # Already embedded
        return False

    def process_file(self, filename: str) -> None:
        """Processes a single JSON file: loads content, generates embeddings, and saves them."""
        if self.is_already_embedded(filename):
            print(f"Skipping {filename} (already embedded).")
            return

        print(f"Processing {filename}")
        filepath = os.path.join(self.input_dir, filename)
        chunks = self.load_json(filepath)

        if not chunks:
            print(f"Skipped {filename} due to loading error.")
            return

        embedded_data = []
        for chunk in tqdm(chunks, desc=f"Embedding {filename}"):
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id", "unknown")
            title = chunk.get("title", "Untitled")
            source = chunk.get("source", "Unknown")

            if not text.strip():
                continue  # Skip empty texts

            embedding = self.generate_embedding(text)
            if embedding:
                embedded_data.append({
                    "chunk_id": chunk_id,
                    "title": title,
                    "text": text,
                    "source": source,
                    "embedding": embedding
                })

        if embedded_data:
            self.save_jsonl(filename, embedded_data)

    def run(self) -> None:
        """Runs the embedding generation process for all JSON files in the input directory, skipping already processed ones."""
        files = [f for f in os.listdir(self.input_dir) if f.endswith(".json")]
        for file in files:
            self.process_file(file)


if __name__ == "__main__":
    embedder = EmbeddingGenerator()
    embedder.run()