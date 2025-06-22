import streamlit as st
import pandas as pd
import numpy as np
import faiss
import torch
import re
from transformers import AutoTokenizer, AutoModel

@st.cache_resource
def load_model():
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def get_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = mean_pooling(outputs, inputs['attention_mask'])
    return embeddings.detach().numpy().astype("float32")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

def highlight(text, query):
    pattern = re.compile(r'(' + '|'.join(map(re.escape, query.split())) + r')', re.IGNORECASE)
    return pattern.sub(r'**\1**', text)

def semantic_search(query, df, tokenizer, model, genre="All", top_k=5):
    if genre != "All" and "genre" in df.columns:
        mask = df["genre"].fillna("").str.contains(genre, case=False)
        sub_df = df[mask]
    else:
        sub_df = df.copy()

    sub_df = sub_df[sub_df["fulltext"].notna()].reset_index(drop=True)
    if sub_df.empty:
        return []

    texts = sub_df["fulltext"].tolist()
    embeddings = get_embeddings(texts, tokenizer, model)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    query_embedding = get_embeddings([query], tokenizer, model)
    distances, indices = index.search(query_embedding, top_k)

    results = sub_df.iloc[indices[0]].copy()
    results["score"] = distances[0]
    return results

# Load resources
tokenizer, model = load_model()
df = pd.read_csv("meta_manga_novel.csv")

# UI
st.title("📚 AI-Powered Novel Search (CPU Safe, Py3.13)")
query = st.text_input("🔍 Search your favorite topic:")
genres = ['All'] + sorted(df['genre'].dropna().unique()) if "genre" in df.columns else ['All']
selected_genre = st.selectbox("🎭 Genre filter", genres)
top_k = st.slider("Number of results", 1, 10, 5)

if query:
    with st.spinner("Searching..."):
        results = semantic_search(query, df, tokenizer, model, genre=selected_genre, top_k=top_k)
        if results.empty:
            st.warning("No results found.")
        else:
            for _, row in results.iterrows():
                st.markdown(f"### {row['title']}")
                st.markdown(f"*{row['summary']}*")
                snippet = row['fulltext'][:300].replace("\n", " ") if pd.notna(row['fulltext']) else ""
                st.markdown(highlight(snippet, query) + "...")
                st.markdown(f"[🔗 Link]({row['link']})")
                st.caption(f"Score: {row['score']:.4f}")
