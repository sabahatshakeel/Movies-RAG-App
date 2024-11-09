import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Initialize the Groq API client
client = Groq(api_key=GROQ_API_KEY)

# Cache the loading of the dataset and embeddings
@st.cache_resource
def load_data():
    return pd.read_csv("movie_dataset.csv")

@st.cache_resource
def load_embeddings():
    return np.load("embeddings.npy")

# Load the movie dataset
# data = pd.read_csv("./movie_dataset.csv")
# data = pd.read_csv(os.path.join(os.path.dirname(__file__), "movie_dataset.csv"))

# Load data and embeddings using the cached functions
data = load_data()
embeddings = load_embeddings()

# Initialize the Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# # Convert movie descriptions to embeddings
# embeddings = model.encode(data['overview'].fillna("").tolist())
# embeddings = np.array(embeddings).astype('float32')  # FAISS requires float32

# Create a FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
index.add(embeddings)

# Define the RAG function for movie query
def query_movie_rag(query):
    # Encode the user query
    query_emb = model.encode([query])[0].astype('float32')
    
    # Search FAISS for top 5 matches
    _, indices = index.search(np.array([query_emb]), k=5)
    
    # Compile the top 5 movies based on similarity
    recommended_movies = [
        f"{data['original_title'][i]}: {data['overview'][i]}"
        for i in indices[0]
    ]
    
    # Format results for output
    response = "\n\n".join([f"{idx+1}. {movie}" for idx, movie in enumerate(recommended_movies)])
    return f"Top recommended movies based on your search:\n\n{response}"

# Streamlit UI setup
st.title("Movie RAG Chatbot")
st.write("Enter a query about movies (e.g., 'funny movies').")

# Text input for user query
user_query = st.text_input("Search for movies:", "")

# Display response when user submits a query
if user_query:
    response = query_movie_rag(user_query)
    st.write(response)
