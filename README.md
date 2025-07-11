# 🧠 Hybrid Search with PGVector

A Streamlit application implementing hybrid semantic search using PGVector. Combines dense OpenAI embeddings and sparse TF-IDF vectors to boost recall and accuracy on financial documents.

---

## 📆 Features

- PGVector-based dense vector search
- Sparse vector (TF-IDF) integration for hybrid results
- Evaluation script to compare Dense vs Hybrid search
- Streamlit interface for interactive querying

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/Vibhorsinha15/pgvector-hybrid-search.git
cd pgvector-hybrid-search
```

### 2. Set up environment

1. Copy and update credentials in `.env`:
```bash
cp .env.example .env
```
2. Fill in your OpenAI API key and Postgres DB credentials.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up PostgreSQL database

Make sure you have PGVector extension enabled:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```
Then run the schema:
```bash
psql -U your_user -d pgvector_db -f db_setup.sql
```

### 5. Load documents
Prepare your corpus (e.g., annual reports, earnings calls) and run:
```bash
python data_loader.py
```

### 6. Launch the app
```bash
streamlit run app.py
```

---

## 📊 Evaluation
To compare performance of Dense vs Hybrid search:
```bash
python evaluation.py
```
Output includes average query time for both approaches.

---

## 📁 Project Structure
```
├── app.py               # Streamlit UI
├── data_loader.py       # Embedding + ingestion
├── db_setup.sql         # SQL schema for documents table
├── evaluation.py        # Performance benchmark
├── requirements.txt     # Dependencies
├── .env.example         # Env template
└── README.md            # You're here!
```

---

## 📹 Demo Video
A walkthrough of the app and results comparison is available in the repo.

---

## 📌 Note
- PGVector must be installed manually (no managed DBs like Supabase for this task).
- Works best with OpenAI `text-embedding-ada-002`.

---

## 🧠 Author
Made with ❤️ by [Vibhor Sinha](https://github.com/Vibhorsinha15)
