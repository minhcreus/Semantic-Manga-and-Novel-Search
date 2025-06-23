import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Novel Search App", layout="wide")
st.title("Semantic Search for Wuxia Novels")

# Load data and embeddings
df = pd.read_csv("wuxia_novel_details_dupli_dropped.csv")
embeddings = np.load("novel_embeddings.npy")

# Load Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Optional filter fields
genres = sorted(df['genre'].dropna().unique()) if 'genre' in df.columns else []
authors = ['All'] + sorted(df['author'].dropna().unique()) if 'author' in df.columns else []
languages = ['All'] + sorted(df['language'].dropna().unique()) if 'language' in df.columns else []
tags = ['All'] + sorted(df['tag'].dropna().unique()) if 'tag' in df.columns else []

# Sidebar filters
with st.sidebar:
    st.header("🔎 Filters")
    selected_genres = st.multiselect("Genres", genres)
    selected_author = st.selectbox("Author", authors)
    selected_language = st.selectbox("Language", languages)
    selected_tag = st.selectbox("Tag", tags)
    top_k = st.slider("Number of results", 1, 20, 5)

# Query input
query = st.text_input("Enter your search query:")

# Filter mask
mask = pd.Series([True] * len(df))
if selected_genres:
    mask &= df['genre'].isin(selected_genres)
if selected_author != 'All':
    mask &= df['author'] == selected_author
if selected_language != 'All':
    mask &= df['language'] == selected_language
if selected_tag != 'All':
    mask &= df['tag'].str.contains(selected_tag, case=False, na=False)

sub_df = df[mask].reset_index(drop=True)
sub_embeddings = embeddings[mask.values]

# Function to highlight matched query
def highlight(text, query):
    pattern = re.compile(r'(' + '|'.join(map(re.escape, query.split())) + r')', re.IGNORECASE)
    return pattern.sub(r'**\1**', text)
    
if query:
    query_embedding = model.encode([query], show_progress_bar=False)
    similarities = cosine_similarity(query_embedding, sub_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    top_matches = sub_df.iloc[top_indices]
    top_similarities = similarities[top_indices]

    for i, (idx, row) in enumerate(top_matches.iterrows()):
        st.markdown(
            f"""
            <div style='background-color:#f8f9fa;padding:10px;border-radius:8px;margin-bottom:10px;'>
                <h4>{i+1}. {row['title']}</h4>
                <b>Author:</b> {row.get('author', 'Unknown')} | 
                <b>Genres:</b> {row.get('genre', 'Unknown')} | 
                <b>Language:</b> {row.get('language', 'Unknown')} | 
                <b>Similarity:</b> {top_similarities[i]:.2f}<br><br>
                {highlight(row.get('Summary', 'No summary available.'), query)}
            </div>
            """, unsafe_allow_html=True)

elif len(sub_df) > 0:
    st.info(f"Showing {len(sub_df)} novels (filtered). Enter a query to search.")
else:
    st.warning("No results match your current filters.")
