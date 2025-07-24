import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import nltk
from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
import os
from datetime import datetime
import logging
import time
import json # Import the json module

# --- Setup Logging for Feedback ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE = os.path.join(LOG_DIR, "search_log.csv")

# Setup a basic logger for application events
logger = logging.getLogger("search_app")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(os.path.join(LOG_DIR, "app_events.log"))
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


def log_feedback(timestamp, query, result_title, score, feedback_type):
    """Appends a feedback entry to the CSV log file."""
    log_entry = {
        "timestamp": [timestamp],
        "query": [query],
        "result_title": [result_title],
        "score": [score],
        "feedback": [feedback_type]
    }
    log_df = pd.DataFrame(log_entry)

    if not os.path.exists(LOG_FILE):
        log_df.to_csv(LOG_FILE, index=False, header=True)
    else:
        log_df.to_csv(LOG_FILE, mode='a', index=False, header=False)

@st.cache_resource
def get_nlp_tools():
    """Downloads NLTK data and returns initialized tools."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    lemmatizer = WordNetLemmatizer()
    tokenizer = TreebankWordTokenizer()
    return lemmatizer, tokenizer

@st.cache_resource
def load_model():
    """Loads the Sentence Transformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_data():
    """Loads the data and embeddings from disk."""
    try:
        df = pd.read_csv("meta_manga_novel_with_genre_standardlized.csv")
        embeddings = np.load("manga_novel_embeddings.npy")
        return df, embeddings
    except FileNotFoundError:
        st.error("Error: `meta_manga_novel_with_genre_standardlized.csv` or `manga_novel_embeddings.npy` not found.")
        st.error("Please ensure the necessary data files are in the same directory.")
        return None, None

@st.cache_data
def load_ground_truth():
    """Loads the ground truth data from a JSON file."""
    try:
        with open("Ground_truth.json", 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.sidebar.error("`Ground_truth.json` not found. Please add it to the directory.")
        return None

def evaluate_results(retrieved_titles, relevant_titles):
    """Calculates Precision, Recall, and F1-Score."""
    retrieved_set = set(retrieved_titles)
    relevant_set = set(relevant_titles)
    
    if not relevant_set: return 0, 0, 0 # Cannot evaluate without relevant titles
    
    true_positives = len(retrieved_set.intersection(relevant_set))
    
    precision = true_positives / len(retrieved_set) if retrieved_set else 0
    recall = true_positives / len(relevant_set)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

# --- Main App ---
st.set_page_config(layout="wide", page_title="Semantic Search Engine")
st.title("Novel & Manga Semantic Search Engine")

# Load resources
model = load_model()
lemmatizer, tokenizer = get_nlp_tools()
df, embeddings = load_data()
ground_truth = load_ground_truth()

if df is None:
    st.stop()

# Prepare data and filters
df.columns = [col.strip().capitalize() for col in df.columns]
has_genre = "Genre" in df.columns
has_type = "Type" in df.columns
unique_genres = sorted(set(g.strip() for g_list in df['Genre'].dropna() for g in str(g_list).split(','))) if has_genre else []
unique_types = df['Type'].dropna().unique().tolist() if has_type else []

def normalize_query(query):
    """Lemmatizes and lowercases the user query."""
    tokens = tokenizer.tokenize(query.lower())
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens if re.match(r'\w+', t)]
    return ' '.join(lemmatized)

def highlight(text, query):
    """Highlights query terms in the result text."""
    if not isinstance(text, str): return ""
    query_words = set(query.lower().split())
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, query_words)) + r')\b', re.IGNORECASE)
    return pattern.sub(r'**\1**', text)

@st.cache_data(show_spinner=False)
def semantic_search(_model, query, df, embeddings, selected_genres=None, type_filter=None, top_k=30):
    """Performs semantic search using FAISS."""
    filtered_df = df.copy()
    if has_type and type_filter and 'All' not in type_filter:
        filtered_df = filtered_df[filtered_df['Type'].isin(type_filter)]
    if has_genre and selected_genres and 'All' not in selected_genres:
        pattern = '|'.join(map(re.escape, selected_genres))
        filtered_df = filtered_df[filtered_df['Genre'].str.contains(pattern, na=False)]

    if filtered_df.empty: return pd.DataFrame()
    
    original_indices = filtered_df.index
    sub_embeddings = embeddings[original_indices]
    sub_df = filtered_df.reset_index(drop=True)
    if sub_df.empty: return pd.DataFrame()

    d = sub_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(sub_embeddings)
    index.add(sub_embeddings.astype('float32'))

    normalized_query = normalize_query(query)
    query_embedding = _model.encode([normalized_query], normalize_embeddings=True).astype('float32')
    distances, indices = index.search(query_embedding, top_k)

    if len(indices[0]) == 0 or indices[0][0] == -1: return pd.DataFrame()

    results_df = sub_df.iloc[indices[0]].copy()
    results_df['Score'] = distances[0]
    return results_df

# --- Sidebar ---
st.sidebar.header("Filters")
selected_types = st.sidebar.multiselect("Filter by Type:", options=['All'] + unique_types, default=['All']) if has_type and unique_types else []
selected_genres = st.sidebar.multiselect("Filter by Genre:", options=['All'] + unique_genres, default=['All']) if has_genre and unique_genres else []

st.sidebar.header("Evaluation")
if ground_truth:
    eval_query = st.sidebar.selectbox("Select a query to evaluate:", list(ground_truth.keys()))
    top_k_eval = st.sidebar.slider("Top-K for Evaluation", min_value=1, max_value=30, value=10)

    if st.sidebar.button("Evaluate Search Results"):
        relevant_titles = ground_truth.get(eval_query, [])
        retrieved_results = semantic_search(model, eval_query, df, embeddings, top_k=top_k_eval)
        
        if not retrieved_results.empty and relevant_titles:
            retrieved_titles = retrieved_results['Title'].tolist()
            precision, recall, f1_score = evaluate_results(retrieved_titles, relevant_titles)
            
            st.sidebar.markdown(f"**Results for '{eval_query}' (Top {top_k_eval})**")
            st.sidebar.markdown(f"**Precision:** `{precision:.3f}`")
            st.sidebar.markdown(f"**Recall:** `{recall:.3f}`")
            st.sidebar.markdown(f"**F1-Score:** `{f1_score:.3f}`")
        else:
            st.sidebar.warning("Could not perform evaluation. No results or no ground truth titles.")
else:
    st.sidebar.info("Evaluation module disabled. `Ground_truth.json` not found.")


# --- Main Search UI ---
query = st.text_input(
    "Enter a query (e.g., apocalypse, reincarnation, cultivation):",
    placeholder="Search for a story about a hero returning to the past..."
)

if query:
    if 'current_query' not in st.session_state or st.session_state.current_query != query:
        st.session_state.current_query = query
        st.session_state.current_page = 0
        logger.info(f"New search: '{query}'")

    start_time = time.time()
    with st.spinner("Searching..."):
        results = semantic_search(model, query, df, embeddings, selected_genres, selected_types, top_k=30)
    end_time = time.time()
    search_duration = end_time - start_time

    if results.empty:
        st.warning("No results found for your query or filter selection.")
    else:
        st.markdown(f"Found **{len(results)}** results. _Search took: **{search_duration:.2f}** seconds._")
        
        results_per_page = 10
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 0
        
        start_idx = st.session_state.current_page * results_per_page
        end_idx = start_idx + results_per_page
        paginated_results = results.iloc[start_idx:end_idx]

        for i, row in paginated_results.iterrows():
            st.markdown("---")
            result_title = row.get('Title', 'No Title')
            st.markdown(f"### {result_title}")

            meta_cols = st.columns(3)
            meta_cols[0].markdown(f"**Similarity Score:** `{row['Score']:.4f}`")
            if pd.notna(row.get("Type")): meta_cols[1].markdown(f"**Type:** `{row['Type']}`")
            if pd.notna(row.get("Link")): meta_cols[2].markdown(f"**Source:** [Read Here]({row['Link']})")
            if pd.notna(row.get("Genre")): st.markdown(f"**Genres:** `{row['Genre']}`")

            summary_text = row.get("Summary", "No summary available.")
            st.markdown(highlight(summary_text, query), unsafe_allow_html=True)

            feedback_key = f"feedback_{i}_{query}"
            if st.session_state.get(feedback_key, False):
                st.success("Thanks for your feedback! 👍")
            else:
                feedback_cols = st.columns([1, 1, 8])
                if feedback_cols[0].button("👍 Like", key=f"like_{i}_{result_title}"):
                    log_feedback(datetime.now(), query, result_title, row['Score'], "like")
                    st.session_state[feedback_key] = True
                    st.rerun()
                if feedback_cols[1].button("👎 Dislike", key=f"dislike_{i}_{result_title}"):
                    log_feedback(datetime.now(), query, result_title, row['Score'], "dislike")
                    st.session_state[feedback_key] = True
                    st.rerun()

        st.markdown("---")
        total_pages = (len(results) + results_per_page - 1) // results_per_page
        if total_pages > 1:
            page_cols = st.columns([1, 1, 10, 1, 1])
            
            if page_cols[0].button("⬅️ Previous", disabled=(st.session_state.current_page == 0)):
                st.session_state.current_page -= 1
                st.rerun()

            page_cols[1].markdown(f"Page **{st.session_state.current_page + 1}** of **{total_pages}**")

            if page_cols[4].button("Next ➡️", disabled=(st.session_state.current_page >= total_pages - 1)):
                st.session_state.current_page += 1
                st.rerun()
else:
    st.info("Enter a query to search for novels and manga.")
