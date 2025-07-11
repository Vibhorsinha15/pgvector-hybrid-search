import os
import openai
import psycopg2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "port": 5432
}

def get_openai_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']

def connect_db():
    return psycopg2.connect(**DB_CONFIG)

# Load all text files from documents folder
def load_documents(folder="documents"):
    docs = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                docs.append(f.read())
    return docs

def insert_documents(texts):
    vectorizer = TfidfVectorizer(max_features=768)
    tfidf_matrix = vectorizer.fit_transform(texts).toarray()
    conn = connect_db()
    cur = conn.cursor()
    for i, text in enumerate(texts):
        embedding = get_openai_embedding(text)
        tfidf_vector = tfidf_matrix[i].tolist()
        cur.execute(
            "INSERT INTO documents (content, embedding, tfidf_vector) VALUES (%s, %s, %s)",
            (text, embedding, tfidf_vector)
        )
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    documents = load_documents()
    insert_documents(documents)
    print(f"Inserted {len(documents)} documents into the database.")
