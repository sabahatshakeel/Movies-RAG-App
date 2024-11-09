import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

# Streamlit caching for model and data
@st.cache_resource
def load_model():
    # Load SentenceTransformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

# @st.cache_data
# def load_data():
#     # Load the movie dataset (e.g., hosted on S3 or locally)
#     if not os.path.exists('./movie_dataset.csv'):
#         # Replace this with actual download if dataset is not local
#         st.error("Movie dataset not found!")
#         return None
#     data = pd.read_csv('./movie_dataset.csv')
#     # data = pd.read_csv(os.path.join(os.path.dirname(__file__), "./movie_dataset.csv"))
#     return data

@st.cache_data
def load_data():
    file_path = "./movie_dataset.csv"
    if not os.path.exists(file_path):
        st.error(f"Movie dataset not found at {movie_dataset.csv}!")
        return None
    data = pd.read_csv("./movie_dataset.csv")
    return data

@st.cache_resource
def get_faiss_index(data, model):
    # Load or create FAISS index
    embeddings = model.encode(data['overview'].fillna("").tolist())
    embeddings = np.array(embeddings).astype('float32')  # FAISS requires float32
    
    # Create FAISS index (Using IndexIVFFlat for faster search)
    index = faiss.IndexIVFFlat(embeddings.shape[1], 100)  # 100 is the number of clusters
    if not index.is_trained:
        index.train(embeddings)  # Train the index if not already trained
    index.add(embeddings)  # Add embeddings to the index
    return index, embeddings

# Main Streamlit app
def main():
    # Load model and data
    model = load_model()
    data = load_data()
    
    if data is None:
        return  # Exit if data is not loaded properly
    
    # Create FAISS index
    index, embeddings = get_faiss_index(data, model)
    
    # Streamlit UI
    st.title("Movie Recommendation System")
    query = st.text_input("Enter movie query:")
    
    if query:
        # Convert query to embedding
        query_embedding = model.encode([query]).astype('float32')
        
        # Perform FAISS search (search top 5 nearest neighbors)
        D, I = index.search(query_embedding, 5)
        
        # Display results
        st.write(f"Top 5 similar movies to '{query}':")
        
        for i in range(5):
            st.write(f"{i+1}. {data.iloc[I[0][i]]['title']} - {data.iloc[I[0][i]]['overview']} (Score: {D[0][i]:.4f})")

if __name__ == "__main__":
    main()
