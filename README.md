# Bleach Wiki RAG Bot

## 1️⃣ Introduction
Bleach is one of the most well-known anime and manga series, filled with an extensive lore, complex character relationships, and powerful abilities. However, finding detailed and accurate information across its vast universe can be time-consuming.

Enter **Bleach Wiki RAG Bot**—a Retrieval-Augmented Generation (RAG) chatbot that allows users to ask detailed questions about Bleach and receive accurate, context-aware answers. Instead of manually searching through wiki pages, users can simply query the bot and retrieve relevant information instantly.

## 2️⃣ Project Overview
This project builds an end-to-end RAG pipeline by:
- **Scraping** structured content from the Bleach Wiki.
- **Processing & Chunking** text for better retrieval.
- **Embedding & Storing** knowledge in a vector database.
- **Retrieving & Generating** accurate responses with a language model.

The system ensures efficient data retrieval, factual consistency, and minimal hallucinations.

### 📁 Directory Structure
```
Bleach_Bot/
├── bleach_wiki/         # Stores all processed data
│   ├── raw/            # Raw scraped pages from Bleach Wiki
│   ├── processed/      # Cleaned and structured JSON files
│   ├── chunks/         # Chunked text data for embeddings
│   ├── embeddings/     # Vector embeddings for retrieval
│   ├── vector_db/      # ChromaDB storage for fast lookup
│
├── src/                # Core RAG pipeline logic
│   ├── scraper/        # Web scraping and preprocessing
│   │   ├── page_collector.py # Extracts all page names for scraping
│   │   ├── scraper.py       # Scrapes content from Bleach Wiki
│   │   ├── preprocess.py    # Cleans and structures raw data
│   │   └── chunker.py       # Splits text into smaller chunks
│   │
│   ├── embedder/       # Embedding logic for retrieval
│   │   ├── embed.py    # Converts text into vector embeddings
│   │   └── vector_store.py  # Stores and retrieves embeddings
│   │
│   ├── pipeline/       # Core retrieval and generation logic
│   │   ├── retrieval.py       # Retrieves relevant information
│   │   ├── retrieval_chain.py # Manages RAG pipeline
│   │   ├── llm.py             # LLM-based response generation
│   │
│   ├── chatbot/        # Handles user queries and interaction
│   │   └── mainbot.py  # Main chatbot logic
│   │
│   ├── evaluation/     # Performance & faithfulness evaluation
│   │   ├── evaluation.py  # Computes faithfulness & accuracy
│   │   └── logging.py     # Logs evaluation results
│   │
│   ├── frontend/       # Streamlit-based UI for chatbot
│   │   ├── streamlit_app1.py  # Streamlit UI Version 1
│   │   ├── streamlit_app2.py  # Streamlit UI Version 2
│
├── README.md           # Documentation & project overview
├── requirements.txt    # Required dependencies
```

---

## 3️⃣ Scraper 
### Overview
The scraper module extracts structured data from Bleach Wiki using **Selenium** and **BeautifulSoup**. The data is then cleaned, chunked, and stored for retrieval.

### 📂 Components

#### 1️⃣ **page_collector.py**
🔹 Collects all character names listed in the Bleach Wiki.
🔹 Uses Selenium to navigate through pages.
🔹 Stores the extracted names for scraping.

#### 2️⃣ **scraper.py**
🔹 Fetches Bleach Wiki pages using Selenium.
🔹 Parses structured content using BeautifulSoup.
🔹 Extracts sections, descriptions, and metadata.
🔹 Saves results as JSON files in `bleach_wiki/raw/`.

#### 3️⃣ **preprocessor.py**
🔹 Cleans and processes raw JSON data by:
   - Removing unnecessary elements.
   - Flattening subsections for structured retrieval.
🔹 Stores results in `bleach_wiki/processed/`.

#### 4️⃣ **chunker.py**
🔹 Splits processed text into smaller, meaningful chunks.
🔹 Ensures coherence for better retrieval.
🔹 Saves chunked text in `bleach_wiki/chunks/`.

---

## 4️⃣ Embedder & Vector Storage
### Overview
The embedding module converts text into high-dimensional vector embeddings using **Google's Gemini Model** and stores them in **ChromaDB** for fast similarity-based retrieval.

### 📂 Components

#### 1️⃣ **Text Embedding with Gemini (embed.py)**
🔹 Converts textual chunks into vector embeddings.
🔹 Saves them in `.jsonl` format in `bleach_wiki/embeddings/`.
🔹 Skips already processed files.

#### 2️⃣ **Storing & Retrieving Embeddings (vector_store.py)**
🔹 Loads precomputed embeddings and indexes them in **ChromaDB**.
🔹 Enables similarity-based retrieval for queries.

🔹 **Execution Flow:**
1️⃣ Run `embed.py` → Generates vector embeddings.
2️⃣ Run `vector_store.py` → Stores embeddings in ChromaDB.
3️⃣ Use `retrieval.py` → Retrieves relevant documents.

---

## 5️⃣ RAG Pipeline 
### Overview
The **Retrieval-Augmented Generation (RAG) pipeline** enhances LLM responses by retrieving contextually relevant documents before generating answers.

### 📂 Components

#### 1️⃣ **Retrieval (retrieval.py)**
🔹 Fetches relevant information from ChromaDB based on semantic similarity.
🔹 Uses vector search to retrieve top-matching chunks.


#### 2️⃣ **Generation (llm.py)**
🔹 Takes retrieved context and constructs a query-specific prompt.
🔹 Generates a response using **Gemini LLM**.


#### 3️⃣ **RAG Pipeline Integration (retrieval_chain.py)**
🔹 Combines retrieval and generation for **seamless interaction**.

---
