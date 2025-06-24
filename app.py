import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer

# Load data
df = pd.read_csv("meta_manga_novel_with_genre.csv")
embeddings = np.load("novel_embeddings_genre.npy")
faiss_index = faiss.read_index("novel_index_genre.faiss")

# Verify column availability
required_cols = {"title", "summary", "link", "genre"}
if not required_cols.issubset(set(df.columns)):
    st.error(f"Missing one of the required columns: {required_cols}")
    st.stop()

# Genre menu
has_genre = "genre" in df.columns
genres = sorted(set(g.strip() for g_list in df['genre'].dropna() for g in g_list.split(','))) if has_genre else []
genres = ['All'] + genres

# Load model (no need for .to(device))
model = SentenceTransformer("all-MiniLM-L6-v2")

# Normalize query for consistent matching
def normalize_query(query):
    query = query.lower()
    query = re.sub(r'[^\w\s]', '', query)
    return query

# Highlight terms
def highlight(text, query):
    pattern = re.compile(r'(' + '|'.join(map(re.escape, query.split())) + r')', re.IGNORECASE)
    return pattern.sub(r'**\1**', text)

# Search logic
def semantic_search(query, df, embeddings, model, index, genre="All", top_k=5):
    if genre != "All" and has_genre:
        mask = df["genre"].fillna("").str.contains(genre, case=False)
        sub_df = df[mask].reset_index(drop=True)
        sub_embeddings = embeddings[mask.values]
    else:
        sub_df = df.copy()
        sub_embeddings = embeddings

    if "summary" not in sub_df.columns:
        return pd.DataFrame()

    sub_df = sub_df[sub_df["summary"].notna()].reset_index(drop=True)
    sub_embeddings = sub_embeddings[:len(sub_df)]

    if sub_df.empty:
        return pd.DataFrame()

    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")

    temp_index = faiss.IndexFlatL2(query_embedding.shape[1])
    temp_index.add(sub_embeddings)
    distances, indices = temp_index.search(query_embedding, top_k)

    results = sub_df.iloc[indices[0]].copy()
    results["score"] = distances[0]
    return results

# Streamlit UI
st.title("📚 Manga & Novel Semantic Search")
query = st.text_input("Enter a query (e.g., action, fantasy, romance):")

selected_genre = st.selectbox("Filter by Genre", genres)
top_k = st.slider("Number of results", 1, 20, 5)

if query:
    clean_query = normalize_query(query)
    results = semantic_search(
        clean_query, df, embeddings, model, faiss_index,
        genre=selected_genre, top_k=top_k
    )

    if results.empty:
        st.warning("No matching results found.")
    else:
        for _, row in results.iterrows():
            st.markdown(f"### {row['title']}")
            if pd.notna(row.get("link")):
                st.markdown(f"[🔗 Read here]({row['link']})")
            st.markdown(f"**Score:** {row['score']:.2f}")
            st.markdown(f"**Genres:** {row['genre']}")
            st.markdown(highlight(row["summary"], query), unsafe_allow_html=True)
            st.markdown("---")
