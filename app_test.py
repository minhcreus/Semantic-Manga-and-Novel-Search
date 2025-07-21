import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import nltk
from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

@st.cache_resource
def get_nlp_tools():
    """Downloads NLTK data and returns initialized lemmatizer and tokenizer."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    lemmatizer = WordNetLemmatizer()
    tokenizer = TreebankWordTokenizer()
    return lemmatizer, tokenizer

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_data():
    """Loads the cleaned data and embeddings from disk."""
    try:
        df = pd.read_csv("meta_manga_novel_with_genre_standardlized.csv")
        embeddings = np.load("manga_novel_embeddings.npy")
        return df, embeddings
    except FileNotFoundError:
        st.error("Error: `cleaned_manga_data.csv` or `manga_novel_embeddings.npy` not found.")
        st.error("Please run the `create_embeddings.py` script first to generate these files.")
        return None, None

model = load_model()
lemmatizer, tokenizer = get_nlp_tools()
df, embeddings = load_data()

if df is None:
    st.stop()

df.columns = [col.strip().capitalize() for col in df.columns]

has_genre = "Genre" in df.columns
unique_genres = sorted(set(g.strip() for g_list in df['Genre'].dropna() for g in str(g_list).split(','))) if has_genre else []

def normalize_query(query):
    """Lemmatizes and lowercases the user query."""
    tokens = tokenizer.tokenize(query.lower())
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens if re.match(r'\w+', t)]
    return ' '.join(lemmatized)

def highlight(text, query):
    """Highlights the query terms in the result text."""
    if not isinstance(text, str):
        return ""
    query_words = map(re.escape, query.split())
    pattern = re.compile(r'(' + '|'.join(query_words) + r')', re.IGNORECASE)
    return pattern.sub(r'**\1**', text)

def semantic_search(query, df, embeddings, model, selected_genres=None, top_k=5):
    """
    Performs semantic search using FAISS for efficiency.
    Filters by genre before searching.
    """
    if has_genre and selected_genres and 'All' not in selected_genres:
        original_indices = df.index[df['Genre'].str.contains('|'.join(map(re.escape, selected_genres)), na=False)]
        if len(original_indices) == 0:
            return pd.DataFrame()
        sub_df = df.loc[original_indices].copy().reset_index(drop=True)
        sub_embeddings = embeddings[original_indices]
    else:
        sub_df = df.copy()
        sub_embeddings = embeddings

    if sub_df.empty or len(sub_embeddings) == 0:
        return pd.DataFrame()

    d = sub_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(sub_embeddings.astype('float32'))

    normalized_query = normalize_query(query)
    query_embedding = model.encode([normalized_query], normalize_embeddings=True).astype('float32')
    
    distances, indices = index.search(query_embedding, top_k)

    if len(indices[0]) == 0 or indices[0][0] == -1:
        return pd.DataFrame()

    results_df = sub_df.iloc[indices[0]].copy()
    results_df['Score'] = distances[0]
    
    return results_df

st.set_page_config(layout="wide")
st.title("Novel & Manga Semantic Search Engine")

query = st.text_input("Enter a query (e.g., apocalypse, reincarnation, cultivation):", placeholder="Search for a story about a hero returning to the past...")

col1, col2 = st.columns([3, 1])
with col1:
    if unique_genres:
        selected_genres = st.multiselect("Filter by Genre:", options=['All'] + unique_genres, default=['All'])
    else:
        selected_genres = []
with col2:
    top_k = st.slider("Number of results:", min_value=1, max_value=20, value=5)

if query:
    results = semantic_search(query, df, embeddings, model, selected_genres=selected_genres, top_k=top_k)

    if results.empty:
        st.warning("No results found for your query or genre selection.")
    else:
        st.write(f"Showing top {len(results)} results:")
        for i, row in results.iterrows():
            st.markdown("---")
            st.markdown(f"### {row.get('Title', 'No Title')}")
            
            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                st.markdown(f"**Similarity Score:** `{row['Score']:.4f}`")
                if pd.notna(row.get("Link")):
                    st.markdown(f"**Source:** [Read Here]({row['Link']})")
            with meta_col2:
                 if pd.notna(row.get("Genre")):
                    st.markdown(f"**Genres:** `{row['Genre']}`")

            summary_text = row.get("Summary", "No summary available.")
            st.markdown(highlight(summary_text, query), unsafe_allow_html=True)
else:
    st.info("Enter a query above to search for novels and manga.")
