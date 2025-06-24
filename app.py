import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer

# Load new data
df = pd.read_csv("meta_manga_novel_with_genre.csv")
embeddings = np.load("manga_novel_embeddings.npy")
index = faiss.read_index("manga_novel_index.faiss")

# Genre menu setup
has_genre = "genre" in df.columns.str.lower()
df.columns = df.columns.str.strip().str.lower()  # Normalize column headers
genres = ['All'] + sorted(set(g.strip() for g_list in df['genre'].dropna() for g in g_list.split(','))) if has_genre else ['All']

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

def highlight(text, query):
    pattern = re.compile(r'(' + '|'.join(map(re.escape, query.split())) + r')', re.IGNORECASE)
    return pattern.sub(r'**\1**', text)

def semantic_search(query, df, embeddings, model, index, genre="All", top_k=5):
    if genre != "All" and has_genre:
        mask = df["genre"].fillna("").str.contains(genre, case=False)
        sub_df = df[mask].reset_index(drop=True)
        sub_indices = np.where(mask.values)[0]
    else:
        sub_df = df.copy()
        sub_indices = np.arange(len(df))

    sub_df = sub_df[sub_df["summary"].notna()].reset_index(drop=True)
    sub_indices = sub_indices[:len(sub_df)]

    if sub_df.empty:
        return []

    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices_found = index.search(query_embedding, top_k)

    result_indices = sub_indices[indices_found[0]]
    results = df.iloc[result_indices].copy()
    results["score"] = distances[0]
    return results

st.title("Manga/Novel Semantic Search")
query = st.text_input("Enter a query (e.g., apocalypse, reincarnation, cultivation):")

selected_genre = st.selectbox("Filter by Genre", genres)
top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)

if query:
    results = semantic_search(query, df, embeddings, model, index, genre=selected_genre, top_k=top_k)
    for i, row in results.iterrows():
        st.markdown(f"### {row['title']}")
        if pd.notna(row.get("link")):
            st.markdown(f"[Read here]({row['link']})")
        st.markdown(f"**Score:** {row['score']:.2f}")
        st.markdown(highlight(row["summary"], query), unsafe_allow_html=True)
        st.markdown("---")
