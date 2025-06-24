import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import nltk
from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

# Load data
df = pd.read_csv("meta_manga_novel_with_genre.csv")
embeddings = np.load("manga_novel_embeddings.npy")

# Ensure alignment between df and embeddings
if len(df) != len(embeddings):
    df = df.iloc[:len(embeddings)].copy()
    embeddings = embeddings[:len(df)]

# Standardize column names
if 'Genre' not in df.columns:
    df.columns = [col.strip().capitalize() for col in df.columns]

# Genre menu setup
has_genre = "Genre" in df.columns
unique_genres = sorted(set(g.strip() for g_list in df['Genre'].dropna() for g in str(g_list).split(','))) if has_genre else []

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")
lemmatizer = WordNetLemmatizer()

def normalize_query(query):
    tokens = nltk.word_tokenize(query.lower())
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens if re.match(r'\w+', t)]
    return ' '.join(lemmatized)

def highlight(text, query):
    pattern = re.compile(r'(' + '|'.join(map(re.escape, query.split())) + r')', re.IGNORECASE)
    return pattern.sub(r'**\1**', text)

def semantic_search(query, df, embeddings, model, selected_genres=None, top_k=5):
    if selected_genres and has_genre and 'All' not in selected_genres:
        genre_mask = df['Genre'].apply(lambda g: any(tag in g for tag in selected_genres) if pd.notna(g) else False)
        sub_df = df[genre_mask].reset_index(drop=True)
        sub_embeddings = embeddings[genre_mask.values]
        if len(sub_embeddings) != len(sub_df):
            min_len = min(len(sub_embeddings), len(sub_df))
            sub_df = sub_df.iloc[:min_len].copy()
            sub_embeddings = sub_embeddings[:min_len]
    else:
        sub_df = df.copy()
        sub_embeddings = embeddings[:len(sub_df)]

    sub_df = sub_df[sub_df["Summary"].notna()].reset_index(drop=True)
    sub_embeddings = sub_embeddings[:len(sub_df)]

    if sub_df.empty:
        return []

    normalized_query = normalize_query(query)
    query_embedding = model.encode([normalized_query], normalize_embeddings=True)[0]
    similarities = np.dot(sub_embeddings, query_embedding)
    top_indices = similarities.argsort()[::-1][:top_k]

    results = sub_df.iloc[top_indices].copy()
    results["score"] = similarities[top_indices]
    return results

# Streamlit UI
st.title("Wuxia Novel Semantic Search")
query = st.text_input("Enter a query (e.g., apocalypse, reincarnation, cultivation):")

selected_genres = st.multiselect("Filter by Genre", options=['All'] + unique_genres, default=['All'])
top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)

if query:
    results = semantic_search(query, df, embeddings, model, selected_genres=selected_genres, top_k=top_k)
    for i, row in results.iterrows():
        st.markdown(f"### {row['Title']}")
        if pd.notna(row.get("Link")):
            st.markdown(f"[Read here]({row['Link']})")
        st.markdown(f"**Score:** {row['score']:.4f}")
        st.markdown(f"**Genres:** {row['Genre']}")
        st.markdown(highlight(row["Summary"], query), unsafe_allow_html=True)
        st.markdown("---")
