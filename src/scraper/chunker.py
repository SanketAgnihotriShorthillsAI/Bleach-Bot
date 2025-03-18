import os
import json
import uuid
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter

class BleachWikiChunker:
    """
    A class to process and chunk JSON documents for retrieval-based applications.
    This class reads preprocessed JSON files, splits the text into smaller chunks,
    and saves the chunks for later use in retrieval-augmented generation (RAG).

    Attributes:
        input_dir (str): Directory containing processed JSON files.
        output_dir (str): Directory to store chunked JSON files.
        chunk_size (int): Maximum size of each text chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
    """
    def __init__(

        self,
        input_dir: str = "bleach_wiki/processed",
        output_dir: str = "bleach_wiki/chunks",
        chunk_size: int = 400,
        chunk_overlap: int = 80,
    ) -> None:
        
        """
        Initializes the BleachWikiChunker with input and output directories,
        chunking parameters, and the text splitter.
        Args:
            input_dir (str): Path to the directory containing JSON files.
            output_dir (str): Path to store chunked JSON files.
            chunk_size (int): Number of characters per chunk.
            chunk_overlap (int): Overlap between consecutive chunks.
        """

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        os.makedirs(self.output_dir, exist_ok=True)

        self.splitter = RecursiveCharacterTextSplitter(

            chunk_size=self.chunk_size,

            chunk_overlap=self.chunk_overlap,

            separators=["\n\n", "\n", ". ", " "]

        )

    def load_json(self, filepath: str) -> Optional[List[Dict]]:
        """Loads a JSON file and returns its contents."""
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def save_chunks(self, filename: str, chunks: List[Dict]) -> None:

        """Saves chunked text data to a JSON file."""

        output_path = os.path.join(self.output_dir, filename)
        try:
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(chunks, file, ensure_ascii=False, indent=4)
            print(f"Chunks saved to {output_path}")
        except Exception as e:
            print(f"Error saving chunks for {filename}: {e}")

    def chunk_document(self, document: List[Dict], filename: str) -> List[Dict]:

        """Splits document sections into smaller chunks.
        The text field in each chunk is prepended with "[file name]-[title]-".
        """

        file_base = os.path.splitext(filename)[0]
        chunks = []
        for section in document:
            content = section.get("content", "")
            title = section.get("title", "Untitled")
            source_url = section.get("source_url", "Unknown")  # Original page URL
            split_texts = self.splitter.split_text(content)
            for chunk in split_texts:
                unique_id = uuid.uuid4().hex[:8]  # Generate a short unique ID

                modified_text = f"{file_base}-{title}-{chunk}"
                chunks.append({
                    "title": title,
                    "chunk_id": f"{title.replace(' ', '_')}_{unique_id}",
                    "text": modified_text,
                    "source_url": source_url
                })
        return chunks

    def is_already_chunked(self, filename: str) -> bool:

        """
        Checks if the file is already chunked by comparing timestamps.
        Args:
            filename (str): The name of the JSON file to check.
        Returns:
            bool: True if already chunked, False otherwise.
        """

        raw_file_path = os.path.join(self.input_dir, filename)
        chunked_file_path = os.path.join(self.output_dir, filename)

        if os.path.exists(chunked_file_path):
            raw_mtime = os.path.getmtime(raw_file_path)
            chunked_mtime = os.path.getmtime(chunked_file_path)
            if chunked_mtime >= raw_mtime:
                return True  # Already chunked
        return False

    def process_file(self, filename: str) -> None:

        """Processes a single JSON file: loads content, chunks it, and saves the chunks."""

        if self.is_already_chunked(filename):
            print(f"Skipping {filename} (already chunked).")
            return
        print(f"Processing {filename}")

        filepath = os.path.join(self.input_dir, filename)
        document = self.load_json(filepath)
        if document is None:
            print(f"Skipped {filename} due to loading error.")
            return

        # Pass the filename into the chunking function

        chunks = self.chunk_document(document, filename)
        self.save_chunks(filename, chunks)

    def run(self) -> None:

        """Processes all JSON files in the input directory, skipping already chunked ones."""

        files = [f for f in os.listdir(self.input_dir) if f.endswith(".json")]
        for file in files:
            self.process_file(file)


if __name__ == "__main__":
    chunker = BleachWikiChunker()
    chunker.run()

 