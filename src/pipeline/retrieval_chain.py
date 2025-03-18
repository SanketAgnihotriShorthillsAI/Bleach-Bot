from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from retrieval import Retriever  
from llm import LLM  # Import the LLM class

class RetrievalChain:
    """
    A class that implements a retrieval-augmented generation (RAG) pipeline.

    This class retrieves relevant documents from a vector database using a retriever
    and generates a response using an LLM.

    Attributes:
        retriever (Retriever): Handles retrieving relevant documents from the vector store.
        llm (LLM): The language model responsible for generating responses.
        qa_chain (RetrievalQA): The RAG-based retrieval and response generation chain.
    """

    def __init__(self, vector_store) -> None:
        """
        Initializes the retrieval-based QA chain.

        Args:
            vector_store (VectorStore): An instance of the vector database.
        """
        self.retriever = Retriever(vector_store)  # Initialize retriever with vector store
        self.llm = LLM()  # Initialize LLM

        # Define a custom prompt template
        prompt_template = PromptTemplate.from_template("""
        Answer the following question based only on the provided context.
        Think step by step before providing a detailed answer.

        <context>
        {context}  # All relevant documents retrieved from the vector store
        </context>

        Question: {input}
        """)

        # Create the RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,  # Use the LLM instance
            retriever=self.retriever.retriever,  # Use retriever instance
            chain_type="stuff",  # Other options: "map_reduce", "refine"
            chain_type_kwargs={"prompt": prompt_template}
        )

    def get_answer(self, query: str) -> str:
        """
        Retrieves the best answer for a given query using retrieval-augmented generation.

        Args:
            query (str): The user's question.

        Returns:
            str: The model's generated response.
        """
        response = self.qa_chain.invoke({"query": query})
        return response["result"]  # Extract the answer

if __name__ == "__main__":
    from Bleach_Bot.src.embedder.vector_store import BleachWikiVectorStore  # Import your custom vector store

    # Initialize vector store
    vector_store = BleachWikiVectorStore()
    
    # Create an instance of the RetrievalChain
    retrieval_chain = RetrievalChain(vector_store)

    # Example query
    query = "Who is Ichigo Kurosaki in Bleach?"
    answer = retrieval_chain.get_answer(query)

    print("Answer:", answer)
