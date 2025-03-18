import os
import time
import streamlit as st
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Any
from streamlit_extras.add_vertical_space import add_vertical_space

# Load environment variables
load_dotenv()

# Set up Gemini API client
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("No API key found! Set GOOGLE_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=api_key)

# Set up ChromaDB client
db_dir = "bleach_wiki/vector_db"  
client = chromadb.PersistentClient(path=db_dir)

# Ensure ChromaDB collection exists
try:
    collection = client.get_collection(name="bleach_wiki_embeddings")
except chromadb.errors.InvalidCollectionException:
    st.error("ChromaDB collection 'bleach_wiki_embeddings' does not exist. Run vector_store.py first.")
    st.stop()

# Streamlit Page Configuration
st.set_page_config(page_title="Bleach Chatbot", layout="wide")

# Custom CSS for Modern UI
st.markdown("""
    <style>
        .main {background-color: #f4f4f4;}
        .chat-container {max-width: 700px; margin: auto;}
        .user-message {background-color: #0078ff; color: white; padding: 10px; border-radius: 10px; margin: 10px 0;}
        .bot-message {background-color: #eee; padding: 10px; border-radius: 10px; margin: 10px 0;}
        .sidebar {background-color: #ffffff; padding: 20px; border-radius: 10px;}
        .typing {color: #999;}
    </style>
""", unsafe_allow_html=True)

st.title("💬 Bleach RAG Chatbot")
st.write("Ask me anything about Bleach! This chatbot is powered by a **Retrieval-Augmented Generation (RAG) pipeline**.")

# Sidebar Customization
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Adjust chatbot settings:")
    response_length = st.slider("Max Response Length", min_value=100, max_value=1000, value=500)
    top_k = st.slider("Number of Retrieved Documents", min_value=1, max_value=10, value=5)
    add_vertical_space(2)
    st.write("**Made by AI Enthusiast 🚀**")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    role_class = "user-message" if message["role"] == "user" else "bot-message"
    with st.chat_message(message["role"]):
        st.markdown(f'<div class="{role_class}">{message["content"]}</div>', unsafe_allow_html=True)


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


def generate_answer(query: str, retrieved_docs: List[Dict[str, Any]], response_length: int) -> str:
    """Generate an answer using Gemini Pro based on retrieved context"""
    if not retrieved_docs:
        return "I don't know. No relevant information found."

    context = "\n".join([doc["text"][:response_length] for doc in retrieved_docs])

    prompt = (
        "You are a knowledgeable assistant, trained on Bleach Wiki. "
        "Go through all the provided context and answer the query. "
        "If the context doesn't contain enough information, say \"I don't know\" "
        "instead of making up an answer by yourself.\n\n"
        f"Context:\n{context}\n\nQuery: {query}\n"
    )

    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    
    # Display typing indicator
    with st.chat_message("assistant"):
        st.markdown('<div class="typing">Typing...</div>', unsafe_allow_html=True)
        time.sleep(2)  # Simulate response time

    response = model.generate_content(prompt)

    return response.text


# User input
query = st.chat_input("Type your question here...")

if query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(f'<div class="user-message">{query}</div>', unsafe_allow_html=True)

    # Retrieve relevant documents
    retrieved_docs = retrieve(query, top_k)

    # Generate AI response
    answer = generate_answer(query, retrieved_docs, response_length)

    # Display AI response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(f'<div class="bot-message">{answer}</div>', unsafe_allow_html=True)
