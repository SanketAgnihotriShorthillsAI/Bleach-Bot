import json
import csv
import os

class EvaluationLogger:
    """
    Logs evaluation results for the Bleach Wiki Bot to JSON and CSV.
    """

    def __init__(self, json_path="src/evaluation/evaluation_results.json", csv_path="src/evaluation/evaluation_results.csv"):
        """
        Initializes paths for JSON and CSV logs.

        Args:
            json_path (str): Path to store evaluation results in JSON format.
            csv_path (str): Path to store evaluation results in CSV format.
        """
        self.json_path = json_path
        self.csv_path = csv_path

        # Ensure evaluation directory exists
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)

        # Create CSV file with headers if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["query", "generated_answer", "retrieval_precision",
                                 "retrieval_recall", "faithfulness_score"])

    def log(self, data):
        """
        Logs evaluation data to both JSON and CSV files.

        Args:
            data (dict): Dictionary containing evaluation metrics.
        """
        self.log_to_json(data)
        self.log_to_csv(data)

    def log_to_json(self, data):
        """
        Appends evaluation data to a JSON file.

        Args:
            data (dict): Evaluation results.
        """
        if os.path.exists(self.json_path):
            with open(self.json_path, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        existing_data.append(data)

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)

    def log_to_csv(self, data):
        """
        Appends evaluation data to a CSV file.

        Args:
            data (dict): Evaluation results.
        """
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                data.get("query", ""),
                data.get("generated_answer", ""),
                data.get("retrieval_precision", ""),
                data.get("retrieval_recall", ""),
                data.get("faithfulness_score", "")
            ])
