import google.generativeai as genai
from typing import List, Dict

class Generator:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-flash-latest")

    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, str]]) -> str:
        if not retrieved_docs:
            return "I don't know. No relevant information found."

        if isinstance(retrieved_docs[0], str):  
            context = "\n".join(retrieved_docs)
        elif isinstance(retrieved_docs[0], dict) and "text" in retrieved_docs[0]:
            context = "\n".join([doc["text"] for doc in retrieved_docs])
        else:  # Handle unexpected format
            raise ValueError("Invalid format for retrieved_docs. Expected list of strings or dictionaries with 'text' key.")

        prompt = (
            f"""
            You are a knowledgeable assistant, trained on Bleach Wiki. 
            Go through all the provided context and answer the query. 
            If the context doesn't contain enough information, say \"I don't know\" 
            instead of making up an answer by yourself.\n\n
            
            Context:
            {context}
            
            Query: {query}"
            """
        )

        response = self.model.generate_content(prompt)
        return response.text
