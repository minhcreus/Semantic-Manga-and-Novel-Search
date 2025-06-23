import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer

df = pd.read_csv("wuxia_novel_details_dupli_dropped.csv")
embeddings = np.load("novel_embeddings.npy") 

has_genre = "Genre" in df.columns
genres = ['All'] + sorted(set(g.strip() for g_list in df['Genre'].dropna() for g in g_list.split(','))) if has_genre else ['All']

model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

tokenizer = None 

def highlight(text, query):
    pattern = re.compile(r'(' + '|'.join(map(re.escape, query.split())) + r')', re.IGNORECASE)
    return pattern.sub(r'**\1**', text)

def semantic_search(query, df, embeddings, model, genre="All", top_k=5):
    if genre != "All" and has_genre:
        mask = df["Genre"].fillna("").str.contains(genre, case=False)
        sub_df = df[mask].reset_index(drop=True)
        sub_embeddings = embeddings[mask.values]
    else:
        sub_df = df.copy()
        sub_embeddings = embeddings

    sub_df = sub_df[sub_df["Summary"].notna()].reset_index(drop=True)
    sub_embeddings = sub_embeddings[:len(sub_df)]

    if sub_df.empty:
        return []

    query_embedding = model.encode([query], convert_to_numpy=True)
    index = faiss.IndexFlatL2(query_embedding.shape[1])
    index.add(sub_embeddings)

    distances, indices = index.search(query_embedding, top_k)
    results = sub_df.iloc[indices[0]].copy()
    results["score"] = distances[0]
    return results

st.title("Wuxia Novel Semantic Search")
query = st.text_input("Enter a query (e.g., apocalypse, reincarnation, cultivation):")

selected_genre = st.selectbox("Filter by Genre", genres)
top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)

if query:
    results = semantic_search(query, df, embeddings, model, genre=selected_genre, top_k=top_k)
    for i, row in results.iterrows():
        st.markdown(f"### {row['Title']}")
        if pd.notna(row.get("Link")):
            st.markdown(f"[Read here]({row['Link']})")
        st.markdown(highlight(row["Summary"], query), unsafe_allow_html=True)
        st.markdown("---")
