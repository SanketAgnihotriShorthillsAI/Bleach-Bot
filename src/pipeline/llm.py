from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional


class LLM:
    """
    Handles interaction with the Gemini LLM.
    """

    def __init__(self) -> None:
        """
        Initializes the LLM model with the specified Gemini model.
        """
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    def generate_response(self, query: str, context: str) -> str:
        """
        Generates an AI response based on the retrieved context.

        Args:
            query (str): The user's question.
            context (str): The retrieved documents or contextual information.

        Returns:
            str: The model's response, or a default message if no response is generated.
        """
        print("Generating response from LLM...")

        # Construct the prompt for Gemini LLM
        prompt = f"""
        "You are a knowledgeable assistant, trained on Bleach Wiki. "
        "Go through all the provided context and answer the query. "
        "If the context doesn't contain enough information, say \"I don't know\" "
        "instead of making up an answer by yourself.\n\n"
        
        <context>
        {context}
        </context>
        
        Question: {query}
        """

        # Invoke the model with the constructed prompt
        response = self.model.invoke(prompt)

        # Return response content or a fallback message if response is empty
        return response.content if response else "No response generated."
