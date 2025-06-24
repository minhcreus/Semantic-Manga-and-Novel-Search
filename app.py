import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import unicodedata
from sentence_transformers import SentenceTransformer

df = pd.read_csv("meta_manga_novel_with_genre.csv")
embeddings = np.load("novel_embeddings_genre.npy")
index = faiss.read_index("novel_index_genre.faiss")

def split_genres(genres_string):
    return [g.strip() for g in str(genres_string).split(",") if g.strip()]

all_genres = sorted({genre for sublist in df['genre'].dropna().map(split_genres) for genre in sublist})
authors = ['All'] + sorted(df['author'].dropna().unique()) if 'author' in df.columns else ['All']
languages = ['All'] + sorted(df['language'].dropna().unique()) if 'language' in df.columns else ['All']

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def normalize_query(query):
    query = query.lower()
    query = re.sub(r'[^\w\s]', '', query)
    query = unicodedata.normalize("NFKD", query)
    return query

def highlight(text, query):
    pattern = re.compile(r'(' + '|'.join(map(re.escape, query.split())) + r')', re.IGNORECASE)
    return pattern.sub(r'**\1**', text)

def semantic_search(query, df, index, embeddings, model, selected_genres, selected_author, selected_language, top_k=5):
    mask = pd.Series([True] * len(df))

    if selected_genres:
        mask &= df['genre'].apply(lambda x: any(g in split_genres(x) for g in selected_genres))

    if selected_author != "All" and 'author' in df.columns:
        mask &= df['author'] == selected_author

    if selected_language != "All" and 'language' in df.columns:
        mask &= df['language'] == selected_language

    sub_df = df[mask].copy()
    sub_df = sub_df[sub_df["Summary"].notna()].reset_index(drop=True)
    sub_embeddings = embeddings[mask.values][:len(sub_df)]

    if sub_df.empty:
        return pd.DataFrame()

    query_embedding = model.encode([normalize_query(query)], convert_to_numpy=True)
    _, indices = index.search(query_embedding, top_k)

    results = sub_df.iloc[indices[0]].copy()
    results["score"] = [np.linalg.norm(query_embedding - emb) for emb in sub_embeddings[indices[0]]]
    return results

st.set_page_config(page_title="Manga Novel Search", layout="wide")
st.title("Semantic Search: Manga & Wuxia Novels")

query = st.text_input("Enter a query (e.g., reincarnation, apocalypse, martial arts):")
top_k = st.slider("Top Results", 1, 20, 5)

with st.sidebar:
    st.header("🔎 Filters")
    selected_genres = st.multiselect("Genres", all_genres)
    selected_author = st.selectbox("Author", authors)
    selected_language = st.selectbox("Language", languages)

# ----------------------
# Display Results
# ----------------------
if query:
    results = semantic_search(
        query,
        df,
        index,
        embeddings,
        model,
        selected_genres,
        selected_author,
        selected_language,
        top_k=top_k,
    )

    if not results.empty:
        for i, row in results.iterrows():
            st.markdown(f"### {i+1}. {row['title']}")
            st.markdown(f"**Genres:** {row.get('genre', 'N/A')}")
            st.markdown(f"**Author:** {row.get('author', 'N/A')}")
            st.markdown(f"**Language:** {row.get('language', 'N/A')}")
            st.markdown(f"**Score (L2):** {row['score']:.2f}")
            if pd.notna(row.get("Link")):
                st.markdown(f"[Read here]({row['Link']})")
            st.markdown(highlight(row["Summary"], query), unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.warning("No results found with current filters.")
else:
    st.info("Enter a query above to search semantically through the dataset.")
