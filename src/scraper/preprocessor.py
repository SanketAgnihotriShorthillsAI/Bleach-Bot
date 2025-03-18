import os
import json
import re
from typing import List, Dict, Optional

class Preprocessor:
    """
    A class to clean, filter, and structure text data from scraped JSON files.

    This class loads raw JSON files, removes unwanted text, filters sections, 
    and structures the data for better retrieval and processing.

    Attributes:
        input_folder (str): Directory containing raw JSON files.
        output_folder (str): Directory to store the cleaned and processed JSON files.
    """

    def __init__(self, input_folder: str = "bleach_wiki/raw", output_folder: str = "bleach_wiki/processed") -> None:
        """
        Initializes the Preprocessor with input and output directories.

        Args:
            input_folder (str): Path to the folder containing raw JSON files.
            output_folder (str): Path to the folder for storing processed JSON files.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def load_json(self, filename: str) -> Optional[Dict]:
        """Loads a JSON file from the input folder."""
        try:
            with open(os.path.join(self.input_folder, filename), "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None

    def save_json(self, filename: str, data: List[Dict]) -> None:
        """Saves processed JSON data to the output folder."""
        try:
            with open(os.path.join(self.output_folder, filename), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"Processed: {filename}")
        except IOError as e:
            print(f"Error saving {filename}: {e}")

    def clean_text(self, text: str) -> str:
        """Cleans text by removing unwanted characters and patterns."""
        if not text:
            return ""

        patterns = [
            r"\[edit\s*\|\s*edit source\]", r"\[hide\]", r"Jump up to:.*",
            r"See also:.*", r"\[.*?\]", r"↑", r"\s+"
        ]

        for pattern in patterns[:-1]:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        text = re.sub(patterns[-1], " ", text).strip()

        return text

    def filter_sections(self, sections: List[Dict]) -> List[Dict]:
        """Filters out unwanted sections based on predefined keywords."""
        unwanted_headings = {
            "Ichigo Kurosaki", "Rukia Kuchiki", "Renji Abarai", "Professional Status", "Personal Status",
            "Zanpakutō", "First Appearance", "Voices", "Contents", "References", "Navigation", "Issues",
            "Trivia", "Quotes"
        }
        cleaned_sections = []

        for section in sections:
            heading = self.clean_text(section.get("heading", ""))
            if not heading or any(uw.lower() in heading.lower() for uw in unwanted_headings):
                continue

            section_text = self.clean_text(section.get("text", ""))
            subsections = [
                {"subheading": self.clean_text(sub.get("subheading", "")), "text": self.clean_text(sub.get("text", ""))}
                for sub in section.get("subsections", [])
                if sub.get("subheading") or sub.get("text")
            ]

            if section_text.strip() or subsections:
                cleaned_sections.append({"heading": heading, "text": section_text, "subsections": subsections})

        return cleaned_sections

    def flatten_sections(self, sections: List[Dict], parent_heading: str = "") -> List[Dict]:
        """Flattens nested sections into a structured format for easier retrieval."""
        flattened = []
        for section in sections:
            heading = section.get("heading", "")
            full_heading = f"{parent_heading} - {heading}" if parent_heading else heading

            section_text = section.get("text", "")
            if section_text:
                flattened.append({"title": full_heading, "content": section_text})

            subsections = section.get("subsections", [])
            flattened.extend(self.flatten_sections(subsections, full_heading))

        return flattened

    def is_already_processed(self, filename: str) -> bool:
        """
        Checks if the file is already processed by comparing timestamps.

        Args:
            filename (str): The name of the JSON file to check.

        Returns:
            bool: True if already processed, False otherwise.
        """
        raw_file_path = os.path.join(self.input_folder, filename)
        processed_file_path = os.path.join(self.output_folder, filename)

        # Check if processed file exists and compare modification times
        if os.path.exists(processed_file_path):
            raw_mtime = os.path.getmtime(raw_file_path)
            processed_mtime = os.path.getmtime(processed_file_path)
            if processed_mtime >= raw_mtime:
                return True  # Already processed
        return False

    def preprocess_file(self, filename: str) -> None:
        """Processes a single JSON file: loads, cleans, filters, and structures the data."""
        if self.is_already_processed(filename):
            print(f"Skipping {filename} (already processed).")
            return

        print(f"Processing {filename}")
        data = self.load_json(filename)

        if data is None:
            print(f"Skipping {filename} due to loading error.")
            return

        page_title = data.get("title", "").strip()
        source_url = data.get("url", "").strip()

        flattened_data = []
        if "sections" in data:
            cleaned_sections = self.filter_sections(data["sections"])
            flattened_data.extend(self.flatten_sections(cleaned_sections))

        for chunk in flattened_data:
            chunk["source"] = source_url

        self.save_json(filename, flattened_data)

    def run(self) -> None:
        """Processes all JSON files in the input folder, skipping already processed ones."""
        files = [f for f in os.listdir(self.input_folder) if f.endswith(".json")]
        for file in files:
            self.preprocess_file(file)


if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.run()
