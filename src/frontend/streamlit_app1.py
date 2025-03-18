import os
import streamlit as st
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Set up Gemini API client
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("No API key found! Set GOOGLE_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=api_key)

# Set up ChromaDB client
db_dir = "bleach_wiki/vector_db"  # Ensure this path is correct
client = chromadb.PersistentClient(path=db_dir)
collection = client.get_collection(name="bleach_wiki_embeddings")

# Streamlit UI
st.set_page_config(page_title="Bleach RAG Chatbot", layout="wide")

st.title("Bleach Wiki Chatbot")
st.write("Ask me anything about Bleach!")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def get_query_embedding(query: str) -> List[float]:
    """Generate query embedding using Gemini"""
    response = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    return response["embedding"]


def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve relevant documents using Gemini-generated query embeddings"""
    query_embedding = get_query_embedding(query)
    
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    retrieved_docs = []
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        retrieved_docs.append({"text": doc, "title": metadata.get("title", "Unknown")})

    return retrieved_docs


def generate_answer(query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    """Generate an answer using Gemini Pro based on retrieved context"""
    if not retrieved_docs:
        return "I don't know. No relevant information found."

    context = "\n".join([doc["text"] for doc in retrieved_docs])

    prompt = (
        "You are a knowledgeable assistant, trained on Bleach Wiki. "
        "Go through all the provided context and answer the query. "
        "If the context doesn't contain enough information, say \"I don't know\" "
        "instead of making up an answer by yourself.\n\n"
        f"Context:\n{context}\n\nQuery: {query}\n"
    )

    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(prompt)

    return response.text


# User input
query = st.chat_input("Type your question here...")

if query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve relevant documents
    retrieved_docs = retrieve(query)

    # Generate AI response
    answer = generate_answer(query, retrieved_docs)

    # Display AI response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
