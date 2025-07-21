import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import nltk
from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

@st.cache_resource
def get_nlp_tools():
    """Downloads NLTK data and returns initialized lemmatizer and tokenizer."""
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
    """Loads the cleaned data and embeddings from disk."""
    try:
        df = pd.read_csv("meta_manga_novel_with_genre_standardlized.csv")
        embeddings = np.load("manga_novel_embeddings.npy")
        return df, embeddings
    except FileNotFoundError:
        st.error("Error: `meta_manga_novel_with_genre_standardlized.csv` or `manga_novel_embeddings.npy` not found.")
        st.error("Please ensure the necessary files are in the same directory and have the correct names.")
        return None, None

# --- Main App ---

# Load resources
model = load_model()
lemmatizer, tokenizer = get_nlp_tools()
df, embeddings = load_data()

if df is None:
    st.stop()

# Clean column names
df.columns = [col.strip().capitalize() for col in df.columns]

# Check for essential columns and prepare filters
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
    """Highlights the query terms in the result text."""
    if not isinstance(text, str):
        return ""
    # Use a set for faster lookups and handle case-insensitivity
    query_words = set(query.lower().split())
    # Create a regex pattern that finds whole words only
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, query_words)) + r')\b', re.IGNORECASE)
    return pattern.sub(r'**\1**', text)

def semantic_search(query, df, embeddings, model, selected_genres=None, type_filter=None, top_k=5):
    """
    Performs semantic search using FAISS. Filters by type and genre before searching.
    """
    filtered_df = df.copy()

    # Apply type filter
    if has_type and type_filter and 'All' not in type_filter:
        # Filter rows where the 'Type' column is in the selected type_filter list
        filtered_df = filtered_df[filtered_df['Type'].isin(type_filter)]

    # Apply genre filter
    if has_genre and selected_genres and 'All' not in selected_genres:
        # Filter rows where any of the selected genres are present in the 'Genre' column
        # The `na=False` ensures that rows with NaN in 'Genre' are excluded
        pattern = '|'.join(map(re.escape, selected_genres))
        filtered_df = filtered_df[filtered_df['Genre'].str.contains(pattern, na=False)]

    # If filters result in an empty dataframe, return early
    if filtered_df.empty:
        return pd.DataFrame()

    # Get the original indices to slice the embeddings array
    original_indices = filtered_df.index
    sub_embeddings = embeddings[original_indices]
    sub_df = filtered_df.reset_index(drop=True)

    if sub_df.empty or len(sub_embeddings) == 0:
        return pd.DataFrame()

    # FAISS search
    d = sub_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # Using Inner Product for similarity
    # Ensure embeddings are normalized for IP to be equivalent to cosine similarity
    faiss.normalize_L2(sub_embeddings)
    index.add(sub_embeddings.astype('float32'))

    normalized_query = normalize_query(query)
    query_embedding = model.encode([normalized_query], normalize_embeddings=True).astype('float32')

    distances, indices = index.search(query_embedding, top_k)

    if len(indices[0]) == 0 or indices[0][0] == -1:
        return pd.DataFrame()

    # Create results dataframe from the filtered sub-dataframe
    results_df = sub_df.iloc[indices[0]].copy()
    results_df['Score'] = distances[0]

    return results_df

# --- Streamlit UI ---

st.set_page_config(layout="wide")
st.title("Novel & Manga Semantic Search Engine")

query = st.text_input(
    "Enter a query (e.g., apocalypse, reincarnation, cultivation):",
    placeholder="Search for a story about a hero returning to the past..."
)

# --- Filters ---
st.sidebar.header("Filters")

# Type Filter (FIXED)
if has_type and unique_types:
    selected_types = st.sidebar.multiselect("Filter by Type:", options=['All'] + unique_types, default=['All'])
else:
    selected_types = []
    if not has_type:
        st.sidebar.warning("A 'Type' column was not found in the data.")

# Genre Filter
if has_genre and unique_genres:
    selected_genres = st.sidebar.multiselect("Filter by Genre:", options=['All'] + unique_genres, default=['All'])
else:
    selected_genres = []
    if not has_genre:
        st.sidebar.warning("A 'Genre' column was not found in the data.")


# Top_K Slider
top_k = st.sidebar.slider("Number of results:", min_value=1, max_value=20, value=5)


# --- Search and Display Results ---
if query:
    # Pass the selected types to the search function
    results = semantic_search(
        query,
        df,
        embeddings,
        model,
        selected_genres=selected_genres,
        type_filter=selected_types,
        top_k=top_k
    )

    if results.empty:
        st.warning("No results found for your query or filter selection.")
    else:
        st.write(f"Showing top {len(results)} results:")
        for i, row in results.iterrows():
            st.markdown("---")
            st.markdown(f"### {row.get('Title', 'No Title')}")

            meta_col1, meta_col2, meta_col3 = st.columns(3)
            with meta_col1:
                st.markdown(f"**Similarity Score:** `{row['Score']:.4f}`")
            with meta_col2:
                 if pd.notna(row.get("Type")):
                    st.markdown(f"**Type:** `{row['Type']}`")
            with meta_col3:
                 if pd.notna(row.get("Link")):
                    st.markdown(f"**Source:** [Read Here]({row['Link']})")


            if pd.notna(row.get("Genre")):
                st.markdown(f"**Genres:** `{row['Genre']}`")

            summary_text = row.get("Summary", "No summary available.")
            st.markdown(highlight(summary_text, query), unsafe_allow_html=True)
else:
    st.info("Enter a query above and use the filters in the sidebar to search for novels and manga.")
