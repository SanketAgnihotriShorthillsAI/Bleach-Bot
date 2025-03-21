from pipeline.retriever import Retriever
from pipeline.generator import Generator

class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

    def run(self, query: str):
        retrieved_docs = self.retriever.retrieve(query)

        print("\nTop Retrieved Chunks:")
        for i, doc in enumerate(retrieved_docs, start=1):
            print(f"{i}. {doc['title']} -> {doc['text'][:200]}...")

        answer = self.generator.generate_answer(query, retrieved_docs)

        print("\nGenerated Answer:")
        print(answer)

if __name__ == "__main__":
    query = input("Ask me a question: ")
    rag_pipeline = RAGPipeline()
    rag_pipeline.run(query)
