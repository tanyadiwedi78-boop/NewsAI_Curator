"""
dashboard.py — Simple Streamlit UI to browse the AI news digest history.
Run with: streamlit run dashboard.py
"""

import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title="AI News Digestor", page_icon="📰", layout="wide")

DIGEST_FILE = Path("output/news_digest.json")

st.title("📰 GenAI News Curator")

if not DIGEST_FILE.exists():
    st.warning("No digest data found yet. Run `content_curator.py` first.")
    st.stop()

with open(DIGEST_FILE, "r", encoding="utf-8") as f:
    all_data = json.load(f)

if not all_data:
    st.info("Digest file exists but has no entries yet.")
    st.stop()

# --- Sidebar: date selector ---
dates = sorted(all_data.keys(), reverse=True)
selected_date = st.sidebar.selectbox("Select date", dates)

articles = all_data[selected_date]

# --- Sidebar: category filter ---
categories = sorted(set(a["category"] for a in articles))
selected_categories = st.sidebar.multiselect("Filter by category", categories, default=categories)

# --- Sidebar: search ---
search_query = st.sidebar.text_input("Search title/summary")

# --- Stats row ---
col1, col2, col3 = st.columns(3)
col1.metric("Total articles", len(articles))
col2.metric("Avg relevance", round(sum(a["relevance_score"] for a in articles) / len(articles), 1))
col3.metric("Categories", len(categories))

st.divider()

# --- Filtered article list ---
filtered = [
    a for a in articles
    if a["category"] in selected_categories
    and (search_query.lower() in a["title"].lower() or search_query.lower() in a["summary"].lower())
]

filtered.sort(key=lambda a: a["relevance_score"], reverse=True)

for a in filtered:
    with st.container(border=True):
        st.markdown(f"**{a['title']}**")
        st.caption(f"{a['source']} · {a['category']} · relevance {a['relevance_score']}/10 · tags: {', '.join(a.get('tags', []))}")
        st.write(a["summary"])
        st.markdown(f"[Read more]({a['url']})")

if not filtered:
    st.info("No articles match your filters.")
