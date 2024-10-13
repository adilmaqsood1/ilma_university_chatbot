import streamlit as st
from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from typing import Optional, List

# Constants
DB_FAISS_PATH = '/content/vectorstore/db_faiss'
IMAGE_PATH = '/content/ilma logo.png'  # Path to the image

# API Key for Groq (Replace with your actual API key)
GROQ_API_KEY = "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"
groq_client = Groq(api_key=GROQ_API_KEY)

# Custom LLM class for Groq
class GroqLLM:
    client: Groq
    model: str = "mixtral-8x7b-32768"  # Adjust the model name as needed

    def __init__(self, client):
        self.client = client

    def generate(self, prompt: str) -> str:
        """Generates response using Groq API based on the provided prompt."""
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stop=None,
            model=self.model,
        )
        return chat_completion.choices[0].message.content

# Initialize GroqLLM
llm = GroqLLM(client=groq_client)

# Function to create a conversational bot
def qa_bot():
    """Builds the conversational bot pipeline using FAISS for retrieval and Groq for LLM."""
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    
    # Load FAISS database
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    def groq_retrieval_chain(query):
        """Fetches relevant documents from FAISS and generates answers using Groq."""
        # Retrieve relevant documents
        result = db.similarity_search(query)
        
        # Create the context from the retrieved documents
        context = "\n".join([doc.page_content for doc in result])
        
        # Generate the answer using Groq LLM
        answer = llm.generate(f"{context}\n\nUser query: {query}")
        return answer, result
    
    return groq_retrieval_chain

# Streamlit App UI
def main():
    # Set the Streamlit layout
    st.set_page_config(page_title="ILMA University Chatbot", layout="centered")

    # Sidebar logo and description
    with st.sidebar:
        image = Image.open(IMAGE_PATH)
        st.image(image, width=230)
        st.write("Your personal assistant for university-related queries")

    # Display the title
    st.title("Welcome to ILMA University Chatbot")

    # Load the FAISS retriever
    retriever = qa_bot()

    # Initialize session state for chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input via chat-style input box
    if prompt := st.chat_input("Ask your question about ILMA University"):
        # Add user's message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from the RAG pipeline (Retrieval + Generation)
        with st.chat_message("assistant"):
            with st.spinner("Fetching response..."):
                response, sources = retriever(prompt)
                st.markdown(response)

            # Add the bot's response to session state
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
