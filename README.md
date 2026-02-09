# RAG Document QnA using LangChain + Groq + Streamlit

This project is a Retrieval Augmented Generation (RAG) based Document Question Answering system that allows users to query research papers (PDFs) and receive accurate, context-based answers using LLMs.

The system retrieves relevant document chunks from the uploaded PDFs and generates responses using Groq's Llama model.

---

## Features

- Ask questions directly from research papers
- Uses RAG (Retriever + LLM) architecture
- FAISS Vector Database for similarity search
- Groq LLM (Llama 3.3 70B Versatile)
- Efficient document embedding with session memory
- Shows retrieved document context for transparency

---

## Tech Stack

- Python
- LangChain
- Groq API
- Streamlit
- FAISS Vector Store
- Ollama Embeddings

---

## Project Architecture

1. Load PDF documents from directory
2. Split documents into smaller chunks
3. Convert chunks into embeddings
4. Store embeddings inside FAISS vector database
5. Retrieve relevant chunks based on user query
6. Pass context + query to LLM
7. Generate accurate response

---

## Folder Structure
RAG Document QnA/
|-research_papers/
|-app.py
|-requirements.txt
|-.gitignore
|-README.md

---


---

## Installation & Setup


### Step 1: Clone Repository
git clone <https://github.com/paranjaysoni/RAG-Document-QnA-Chatbot.git>
cd RAG-Document-QnA

### Step 2: Create Virtual Environment
python3 -m venv venv
source venv/bin/activate

### Step 3: Install Dependencies
pip install -r requirements.txt

### Step 4: Add Your GROQ API Key
Create a `.env` file:
GROQ_API_KEY = your_api_key_here

---

## Run the Application
streamlit run app.py


## How It Works

- Click **Document Embedding** button
- System creates vector database from PDFs
- Enter query related to research paper
- Model retrieves relevant content and answers

---

## Key Concepts Used

- Retrieval Augmented Generation (RAG)
- Vector Embeddings
- Semantic Search
- Prompt Engineering
- Document Chunking

---

## Future Improvements

- Save FAISS index locally (persistent storage)
- Add multi-PDF upload support
- Improve UI/UX
- Add conversation memory
- Deploy on HuggingFace / Streamlit Cloud

---

## Author

**Paranjay Soni**

---

## License

This project is for educational and research purposes.
