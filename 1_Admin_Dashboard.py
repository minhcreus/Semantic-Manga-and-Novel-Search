import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(layout="wide", page_title="Admin Dashboard")
st.title("📊 Admin Dashboard: Search Metrics")

# --- Load Log Data ---
LOG_FILE = os.path.join("logs", "search_log.csv")

@st.cache_data(ttl=60)  # Cache data for 60 seconds
def load_log_data():
    """Loads log data and caches it for 60 seconds to improve performance."""
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()

log_df = load_log_data()

if log_df.empty:
    st.warning("No feedback data has been logged yet. Perform searches and provide feedback in the main app.")
    st.stop()

# --- Display Metrics ---
st.header("Overall Metrics")

col1, col2, col3, col4 = st.columns(4)
total_feedback = len(log_df)
unique_queries = log_df['query'].nunique()
likes = log_df[log_df['feedback'] == 'like'].shape[0]
dislikes = log_df[log_df['feedback'] == 'dislike'].shape[0]

col1.metric("Total Feedback Events", total_feedback)
col2.metric("Unique Queries with Feedback", unique_queries)
col3.metric("Total Likes 👍", likes)
col4.metric("Total Dislikes 👎", dislikes)

st.markdown("---")

# --- Charts and Analysis ---
st.header("Search Analysis")

# Top 10 Most Frequent Queries
st.subheader("Top 10 Most Frequent Queries")
top_queries = log_df['query'].value_counts().nlargest(10)
if not top_queries.empty:
    st.bar_chart(top_queries)
else:
    st.info("Not enough query data to display.")

# Feedback Over Time
st.subheader("Feedback Activity Over Time")
log_df['date'] = log_df['timestamp'].dt.date
feedback_over_time = log_df.groupby('date')['timestamp'].count()
if len(feedback_over_time) > 1:
    st.line_chart(feedback_over_time)
else:
    st.info("Not enough time-series data to display.")

# Like vs. Dislike Ratio for Top Queries
st.subheader("Like vs. Dislike Ratio for Top Queries")
feedback_by_query = log_df.groupby(['query', 'feedback']).size().unstack(fill_value=0)
if not feedback_by_query.empty:
    feedback_by_query['total'] = feedback_by_query.sum(axis=1)
    top_feedback_queries = feedback_by_query.sort_values('total', ascending=False).head(10)
    
    fig = px.bar(
        top_feedback_queries[['like', 'dislike']],
        title="Feedback for Top 10 Queries",
        labels={'value': 'Number of Feedbacks', 'query': 'Search Query'},
        color_discrete_map={'like': '#2ECC71', 'dislike': '#E74C3C'}
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No feedback ratios to display.")

if st.checkbox("Show Raw Feedback Data"):
    st.subheader("Raw Log Data")
    st.dataframe(log_df.sort_values('timestamp', ascending=False))