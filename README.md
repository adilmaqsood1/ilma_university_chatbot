# **ILMA University Chatbot 🚀**
This project is an AI-powered chatbot designed to assist users with university-related queries, specifically for ILMA University. Built with the Groq API, Streamlit, LangChain, FAISS, and HuggingFace embeddings, it integrates a large language model, Mixtral-8x7b-32768, to generate highly relevant responses.

# Features ✨
- ### Conversational AI: 
     The chatbot is capable of understanding and responding to various questions related to ILMA University.
- ### Retrieval-Augmented Generation (RAG): 
     By fetching relevant information from a custom knowledge base, the chatbot generates concise and accurate answers.
- ### FAISS for Efficient Search: 
     Utilizes FAISS for fast similarity searches, allowing efficient retrieval of information.
- ### Custom Embeddings: 
     The chatbot uses HuggingFace sentence-transformer embeddings for text search and document retrieval.
- ### Streamlit UI: 
     A user-friendly interface built with Streamlit for a seamless chatbot experience.

# Tech Stack 💻
- ### Streamlit: 
    For building the frontend UI of the chatbot.
- ### Groq API: 
    Powers the chatbot with the Mixtral-8x7b-32768 model.
- ### LangChain: 
    Integrates the conversational AI logic and retrieval system.
- ### FAISS: 
    Efficient vector search for document retrieval.
- ### HuggingFace: 
    Sentence-transformer embeddings for text search and representation.

# How It Works 🔍

### Preprocessing:
  - PDFs are loaded and split into smaller chunks of text using the RecursiveCharacterTextSplitter.
  - Embeddings are generated for each chunk using the HuggingFace all-MiniLM-L6-v2 model.
  - These embeddings are indexed in FAISS for similarity search.

### Query Processing:
  - User input is passed through the retrieval chain, which searches the indexed documents and generates context.
  - The Groq LLM (Mixtral-8x7b-32768) generates a response based on the retrieved context and user query.

### Response Generation:
  - The chatbot generates a response by combining retrieved document data with the large language model’s capabilities.
  - How to Run Locally 🖥️
  - Prerequisites
  - Python 3.10+
  - Streamlit Cloud or local setup
  - Groq API Key

 # Install Dependencies
>  ` pip install -r requirements.txt `

# Potential Issues & Warnings ⚠️
### Hallucination: 
  - Due to the limited knowledge base, the chatbot may generate inaccurate or overly creative responses (hallucinations). Improving the dataset will reduce these occurrences.
### File Paths: 
  - Ensure all file paths (like the FAISS index and image files) are correctly set, especially if deploying on cloud platforms like Streamlit Cloud.

# Future Improvements 🔮
  - Expanding the knowledge base to reduce hallucinations.
  - Enhancing the accuracy of document retrieval.
  - Incorporating more advanced embeddings and fine-tuning the language model.
