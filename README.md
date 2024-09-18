# RAG-ChatPDF-Ollama-Local

**Application UI**
![RAG](https://github.com/TGouriSankar/RAG-ChatPDF-Ollama-Local/blob/main/RAG.png)

---
**Project Workflow**
This section outlines the process of handling PDF data, generating vector embeddings using the nomic-embed-text model, storing them in Chroma, and querying based on user questions.
1. PDF Chunking and Text Splitting
2. Embedding Generation using Ollama’s nomic-embed-text
3. Storing Vectors in Chroma Vector Database
4. Retrieving Answers Based on User Questions By Llama 3.1 Model

---

**Features**

  - **Text Chunking:** Split documents into smaller chunks for embedding using RecursiveCharacterTextSplitter.
  - **Embeddings with Ollama:** Generate embeddings for text chunks using the OllamaEmbeddings model.
  - **Vector Storage in Chroma:** Store embeddings in a Chroma vector store for fast retrieval.

**Prerequisites**

- Ensure you have the following installed on your system before proceeding with the setup:

    - Python 3.9+
    - Docker (for running ollama)

---
**Running the Streamlit application**

1. Clone repo: Run this in your terminal

       git clone https://github.com/TGouriSankar/RAG-ChatPDF-Ollama-Local.git

2. Install Dependencies: Execute to install dependencies

       pip install -r requirements.txt

3. Launch the App: Run to start the Streamlit interface on localhost

       streamlit run streamlit_app.py

---
**License**

This project is licensed under the MIT License
