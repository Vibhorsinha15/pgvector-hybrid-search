import time
import random
from app import hybrid_search, get_dense_embedding, connect_db

TEST_QUERIES = [
    "company revenue trends",
    "capital expenditure for the year",
    "future business outlook",
    "management discussion",
    "quarterly performance overview"
]

TOP_K = 5

def evaluate():
    results = {"hybrid": [], "dense": []}
    conn = connect_db()
    cur = conn.cursor()

    for query in TEST_QUERIES:
        print(f"Evaluating: {query}")

        # Hybrid Search
        start = time.time()
        hybrid_res = hybrid_search(query, top_k=TOP_K)
        hybrid_time = time.time() - start
        results["hybrid"].append(hybrid_time)

        # Dense-only search
        dense_vector = get_dense_embedding(query)
        start = time.time()
        cur.execute("""
            SELECT id, content, (1 - (embedding <#> %s)) AS score
            FROM documents
            ORDER BY score DESC
            LIMIT %s
        """, (dense_vector, TOP_K))
        _ = cur.fetchall()
        dense_time = time.time() - start
        results["dense"].append(dense_time)

    cur.close()
    conn.close()

    avg_hybrid = sum(results["hybrid"]) / len(results["hybrid"])
    avg_dense = sum(results["dense"]) / len(results["dense"])

    print("\n--- Evaluation Results ---")
    print(f"Average Hybrid Search Time: {avg_hybrid:.4f} sec")
    print(f"Average Dense-only Search Time: {avg_dense:.4f} sec")
    print("(You can extend this to calculate precision/recall using labeled results)")

if __name__ == "__main__":
    evaluate()
