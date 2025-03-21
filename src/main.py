from pipeline.pipeline import RAGPipeline

if __name__ == "__main__":
    query = input("Ask me a question: ")
    rag_pipeline = RAGPipeline()
    rag_pipeline.run(query)
