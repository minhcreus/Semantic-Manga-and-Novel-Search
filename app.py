import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import torch
from sentence_transformers import SentenceTransformer, util

# Load data and embeddings
df = pd.read_csv("meta_manga_novel_with_genre.csv")
embeddings = np.load("novel_embeddings_genre.npy")

# Normalize column names for consistency
df.columns = df.columns.str.lower()

# Extract filter options
genres = sorted(set(g.strip() for glist in df['genre'].dropna() for g in glist.split(','))) if 'genre' in df.columns else []
authors = ['All'] + sorted(df['author'].dropna().unique()) if 'author' in df.columns else ['All']
languages = ['All'] + sorted(df['language'].dropna().unique()) if 'language' in df.columns else ['All']
tags = sorted(set(t.strip() for tlist in df['tag'].dropna() for t in tlist.split(','))) if 'tag' in df.columns else []
demographics = ['All'] + sorted(df['demographic'].dropna().unique()) if 'demographic' in df.columns else ['All']
statuses = ['All'] + sorted(df['status'].dropna().unique()) if 'status' in df.columns else ['All']

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Highlight matched query terms
def highlight(text, query):
    pattern = re.compile(r'(' + '|'.join(map(re.escape, query.split())) + r')', re.IGNORECASE)
    return pattern.sub(r'**\1**', text)

# Normalize query
def normalize_query(query):
    return re.sub(r'[\s]+', ' ', query.strip().lower())

# Apply filters
def apply_filters(df, selected):
    mask = pd.Series([True] * len(df))

    if selected['genres']:
        mask &= df['genre'].apply(lambda x: all(g.lower() in x.lower() for g in selected['genres']) if pd.notna(x) else False)
    if selected['include_tags']:
        mask &= df['tag'].apply(lambda x: all(t.lower() in x.lower() for t in selected['include_tags']) if pd.notna(x) else False)
    if selected['exclude_tags']:
        mask &= df['tag'].apply(lambda x: all(t.lower() not in x.lower() for t in selected['exclude_tags']) if pd.notna(x) else True)
    if selected['author'] != 'All':
        mask &= df['author'] == selected['author']
    if selected['language'] != 'All':
        mask &= df['language'] == selected['language']
    if selected['demographic'] != 'All':
        mask &= df['demographic'] == selected['demographic']
    if selected['status'] != 'All':
        mask &= df['status'] == selected['status']

    return df[mask].reset_index(drop=True)

# Semantic search using cosine similarity with SentenceTransformer
def semantic_search(query, filtered_df, model, top_k, full_embeddings, base_df):
    filtered_df = filtered_df[filtered_df['summary'].notna()].reset_index(drop=True)
    index_map = base_df['title'].isin(filtered_df['title'])
    subset_embeddings = full_embeddings[index_map.values][:len(filtered_df)]

    if len(filtered_df) == 0:
        return filtered_df

    query_embed = model.encode([normalize_query(query)], convert_to_tensor=True)
    subset_embeddings_tensor = torch.tensor(subset_embeddings)
    scores = util.cos_sim(query_embed, subset_embeddings_tensor)[0].cpu().numpy()

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = filtered_df.iloc[top_indices].copy()
    results['score'] = scores[top_indices]
    return results

# Streamlit UI
st.set_page_config(page_title="📚 MangaDex-style Novel Search", layout="wide")
st.title("📖 Advanced Manga Novel Search")

query = st.text_input("Enter your query (e.g. reincarnation, demons, sect):")

with st.sidebar:
    st.header("🔍 Advanced Filters")
    selected_genres = st.multiselect("Genres", genres)
    selected_include_tags = st.multiselect("Include Tags", tags, key="include")
    selected_exclude_tags = st.multiselect("Exclude Tags", tags, key="exclude")
    selected_author = st.selectbox("Author", authors)
    selected_language = st.selectbox("Language", languages)
    selected_demo = st.selectbox("Demographic", demographics)
    selected_status = st.selectbox("Status", statuses)
    top_k = st.slider("Number of results", 1, 30, 10)
    sort_by = st.selectbox("Sort by", ["Similarity", "Title", "Author"])

filters = {
    'genres': selected_genres,
    'include_tags': selected_include_tags,
    'exclude_tags': selected_exclude_tags,
    'author': selected_author,
    'language': selected_language,
    'demographic': selected_demo,
    'status': selected_status
}

filtered_df = apply_filters(df, filters)

if query:
    results = semantic_search(query, filtered_df, model, top_k, embeddings, df)

    if sort_by == "Title":
        results = results.sort_values("title")
    elif sort_by == "Author":
        results = results.sort_values("author")
    else:
        results = results.sort_values("score", ascending=False)

    for i, row in results.iterrows():
        st.markdown(f"### {row['title']}")
        if pd.notna(row.get("link")):
            st.markdown(f"[🔗 Read here]({row['link']})")
        st.markdown(f"**Author:** {row.get('author', 'N/A')}  ")
        st.markdown(f"**Genres:** {row.get('genre', 'N/A')}  ")
        st.markdown(f"**Tags:** {row.get('tag', 'N/A')}  ")
        st.markdown(f"**Language:** {row.get('language', 'N/A')}  ")
        st.markdown(f"**Demographic:** {row.get('demographic', 'N/A')}  ")
        st.markdown(f"**Status:** {row.get('status', 'N/A')}  ")
        st.markdown(f"**Similarity Score:** {row['score']:.2f}")
        st.markdown(highlight(row['summary'], query), unsafe_allow_html=True)
        st.markdown("---")

elif len(filtered_df) > 0:
    st.info(f"Showing {len(filtered_df)} filtered novels. Enter a query to perform semantic search.")
else:
    st.warning("⚠️ No novels match the selected filters. Please adjust them.")
