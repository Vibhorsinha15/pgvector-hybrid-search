import os
import openai
import streamlit as st
import psycopg2
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "port": 5432
}

def connect_db():
    return psycopg2.connect(**DB_CONFIG)

def get_dense_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

def hybrid_search(query, top_k=5):
    dense_vector = get_dense_embedding(query)
    conn = connect_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, content,
            (1 - (embedding <#> %s)) AS dense_score,
            (1 - (tfidf_vector <#> %s)) AS sparse_score,
            ((1 - (embedding <#> %s)) + (1 - (tfidf_vector <#> %s))) / 2 AS hybrid_score
        FROM documents
        ORDER BY hybrid_score DESC
        LIMIT %s;
    """, (dense_vector, dense_vector, dense_vector, dense_vector, top_k))

    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

st.set_page_config(page_title="Hybrid PGVector Search", layout="wide")
st.title("🔍 Hybrid Search with PGVector")

query = st.text_input("Enter your query:")
if query:
    with st.spinner("Searching..."):
        results = hybrid_search(query)
        st.subheader("Top Results")
        for idx, (doc_id, content, dense_score, sparse_score, hybrid_score) in enumerate(results, 1):
            st.markdown(f"**Result {idx}**")
            st.markdown(f"**Hybrid Score:** {hybrid_score:.4f} | **Dense:** {dense_score:.4f} | **Sparse:** {sparse_score:.4f}")
            st.markdown(f"> {content[:500]}...")
            st.markdown("---")
