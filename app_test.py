import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import torch
from sentence_transformers import SentenceTransformer, util
from difflib import get_close_matches
from transformers import BertTokenizer, BertModel

# Load BERT for auxiliary query enhancement
tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
model_bert = BertModel.from_pretrained("bert-base-uncased")

# Load new data
df = pd.read_csv("meta_manga_novel_with_genre.csv")
embeddings = np.load("manga_novel_embeddings.npy")

# Normalize column headers
df.columns = df.columns.str.strip().str.lower()
has_genre = "genre" in df.columns

# Genre menu setup
genres = ['All'] + sorted(set(g.strip() for g_list in df['genre'].dropna() for g in g_list.split(','))) if has_genre else ['All']

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def highlight(text, query):
    pattern = re.compile(r'(' + '|'.join(map(re.escape, query.split())) + r')', re.IGNORECASE)
    return pattern.sub(r'**\1**', text)

def normalize_query(query, title_list):
    if not isinstance(query, str):
        return query
    title_list = [str(t) for t in title_list if isinstance(t, str)]
    close_match = get_close_matches(query, title_list, n=1, cutoff=0.85)
    return close_match[0] if close_match else query

def enrich_query_with_bert(raw_query):
    inputs = tokenizer_bert(raw_query, return_tensors="pt")
    outputs = model_bert(**inputs)
    last_hidden_states = outputs.last_hidden_state
    keywords = tokenizer_bert.convert_ids_to_tokens(inputs["input_ids"][0])
    return " ".join([kw for kw in keywords if kw.isalpha() and len(kw) > 2])

def semantic_search(query, df, embeddings, model, genre="All", top_k=5):
    if genre != "All" and has_genre:
        mask = df["genre"].fillna("").str.contains(genre, case=False)
        sub_df = df[mask].reset_index(drop=True)
        sub_embeddings = embeddings[mask.values]
    else:
        sub_df = df.copy()
        sub_embeddings = embeddings

    sub_df = sub_df[sub_df["summary"].notna()].reset_index(drop=True)
    sub_embeddings = sub_embeddings[:len(sub_df)]

    if sub_df.empty:
        return []

    corrected_query = normalize_query(query, sub_df["title"].tolist())
    enriched_query = enrich_query_with_bert(corrected_query)

    query_embedding = model.encode(enriched_query, convert_to_tensor=True)
    corpus_embeddings = torch.tensor(sub_embeddings)
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = sub_df.iloc[top_indices].copy()
    results["score"] = scores[top_indices]
    return results

st.title("Manga/Novel Semantic Search")
query = st.text_input("Enter a query (e.g., apocalypse, reincarnation, cultivation):")

selected_genre = st.selectbox("Filter by Genre", genres)
top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)

if query:
    st.caption("🔍 Xử lí query: Kiểm tra chính tả, khớp với tên title nếu gần đúng, và enrich với BERT")
    st.caption("🎯 Filter theo thể loại: " + selected_genre)
    results = semantic_search(query, df, embeddings, model, genre=selected_genre, top_k=top_k)
    for i, row in results.iterrows():
        st.markdown(f"### {row['title']}")
        if pd.notna(row.get("link")):
            st.markdown(f"[Read here]({row['link']})")
        st.markdown(f"**Score:** {row['score']:.2f}")
        st.markdown(highlight(row["summary"], query), unsafe_allow_html=True)
        st.markdown("---")
