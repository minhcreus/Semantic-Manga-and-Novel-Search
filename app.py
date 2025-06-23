import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling

@st.cache_resource
def load_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    word_embedding_model = Transformer(model_name)
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

@st.cache_data
def load_data():
    df = pd.read_csv("meta_manga_novel.csv")
    embeddings = np.load("novel_embeddings.npy")
    return df, embeddings

def get_embeddings(texts, tokenizer, model):
    return model.encode(texts, convert_to_numpy=True)

def highlight(text, query):
    pattern = re.compile(r'(' + '|'.join(map(re.escape, query.split())) + r')', re.IGNORECASE)
    return pattern.sub(r'**\1**', text)


def semantic_search(query, df, embeddings, tokenizer, model, genre="All", top_k=5):
    if genre != "All" and "genre" in df.columns:
        mask = df["genre"].fillna("").str.contains(genre, case=False)
        sub_df = df[mask].reset_index(drop=True)
        sub_embeddings = embeddings[mask.values]
    else:
        sub_df = df.copy()
        sub_embeddings = embeddings

    sub_df = sub_df[sub_df["fulltext"].notna()].reset_index(drop=True)
    sub_embeddings = sub_embeddings[:len(sub_df)]

    if sub_df.empty:
        return pd.DataFrame()

    query_embedding = get_embeddings([query], tokenizer, model)
    index = faiss.IndexFlatL2(query_embedding.shape[1])
    index.add(sub_embeddings)

    distances, indices = index.search(query_embedding, top_k)
    results = sub_df.iloc[indices[0]].copy()
    results["score"] = distances[0]
    return results

st.set_page_config(page_title="Semantic Novel Search", layout="wide")

st.title("📖 Semantic Search over Manga/Novel Metadata")

query = st.text_input("🔍 Enter your search query:", "")
df, embeddings = load_data()
model, tokenizer = load_model()

genres = ['All'] + sorted(df['genre'].dropna().unique()) if "genre" in df.columns else ['All']
selected_genre = st.selectbox("🎭 Filter by Genre (optional):", genres)
top_k = st.slider("🔢 Number of results:", min_value=1, max_value=20, value=5)

if query:
    with st.spinner("Searching..."):
        results = semantic_search(query, df, embeddings, tokenizer, model, genre=selected_genre, top_k=top_k)

    if not results.empty:
        for _, row in results.iterrows():
            st.markdown(f"### {row.get('title', 'No Title')}")
            st.markdown(f"**Genre**: {row.get('genre', 'N/A')}")
            st.markdown(highlight(str(row.get("fulltext", "")), query), unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.warning("No results found.")
else:
    st.info("Enter a query to begin searching.")
