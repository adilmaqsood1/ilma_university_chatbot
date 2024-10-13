import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = '/content/drive/MyDrive/Colab Notebooks/model/extracted_data.pdf'
DB_FAISS_PATH = '/content/vectorstore/db_faiss'

# Create vector database
def create_vector_db():
    # Use PyPDFLoader directly for a single PDF file
    loader = PyPDFLoader(DATA_PATH)

    # Load the document
    documents = loader.load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Generate embeddings
    embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    
    # Generate embeddings for all texts (use `embed_documents` method to get embeddings for multiple documents)
    texts_embeddings = embeddings_model.embed_documents([text.page_content for text in texts])

    # Create FAISS vectorstore using list of embeddings and original texts
    db = FAISS.from_documents(texts, embeddings_model)

    # Save the FAISS index locally
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()
