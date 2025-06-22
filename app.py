import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

df = pd.read_csv(os.getenv("CSV_PATH", "meta_manga_novel.csv"))
embeddings = np.load("novel_embeddings.npy")
index = faiss.read_index("novel_index.faiss")

if "genre" in df.columns:
    genres = ['All'] + sorted(df['genre'].dropna().unique().tolist())
    has_genre = True
else:
    genres = ['All']
    has_genre = False

def highlight(text, query):
    pattern = re.compile(r'(' + '|'.join(map(re.escape, query.split())) + r')', re.IGNORECASE)
    return pattern.sub(r'****', text)

def semantic_search(query, genre_filter="All", top_k=5):
    mask = df['fulltext'].notna()
    if genre_filter != "All":
        mask &= df['genre'].fillna("").str.contains(genre_filter, case=False)

    sub_df = df[mask].reset_index(drop=True)
    if sub_df.empty:
        return []

    sub_embeddings = model.encode(sub_df['fulltext'].tolist())
    sub_embeddings = np.array(sub_embeddings).astype("float32")

    temp_index = faiss.IndexFlatL2(sub_embeddings.shape[1])
    temp_index.add(sub_embeddings)

    query_vec = model.encode([query]).astype("float32")
    distances, indices = temp_index.search(query_vec, top_k)

    results = sub_df.iloc[indices[0]].copy()
    results["score"] = distances[0]
    return results

st.title("Semantic Manga/Novel Search")
st.markdown("Enter a natural language query to search across a library of novels.")

query = st.text_input("Search Query")

if has_genre:
    genre = st.selectbox("Genre Filter", genres)
else:
    genre = "All"

top_k = st.slider("Top K Results", min_value=1, max_value=10, value=5)

if st.button("Search") and query.strip():
    with st.spinner("Searching..."):
        results = semantic_search(query, genre_filter=genre, top_k=top_k)

        if len(results) == 0:
            st.warning("No results found.")
        else:
            for _, row in results.iterrows():
                st.markdown(f"### {row['title']}")
                st.markdown(f"*{row['summary']}*")
                snippet = row['fulltext'][:300].replace("\n", " ") if pd.notna(row['fulltext']) else ""
                st.markdown(highlight(snippet, query) + "...")
                st.markdown(f"[Link]({row['link']})")
                st.caption(f"Score: {row['score']:.4f}")
else:
    st.info("Enter a query and click Search to begin.")
