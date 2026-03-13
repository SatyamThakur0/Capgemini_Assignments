# Scenario: E‑Commerce Product Search
# - Problem: Shoppers often type queries like “cheap running shoes” or “affordable sneakers for jogging”, but the catalog lists products as “Budget-friendly athletic footwear”. Traditional keyword search misses these connections because customers use everyday language while product listings use marketing terms.
# - Solution: Use embeddings to capture semantic meaning. Embeddings understand that “cheap running shoes” and “budget-friendly athletic footwear” are related concepts. This ensures customers find the right products even when their wording doesn’t match the catalog descriptions.

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

products = [
    "Budget-friendly athletic footwear",
    "Professional running shoes for athletes",
    "Lightweight jogging sneakers",
    "Premium leather formal shoes",
    "Casual everyday sneakers"
]

query = "cheap running shoes"

embeddings = model.encode(products)
query_embeddings = model.encode([query])

similarity = model.similarity(embeddings, query_embeddings)

best_match_index = np.argmax(similarity)


print('='*56)
for i, score in enumerate(similarity):
    print(f'|| {products[i]:<40} --> {score[0]:.3f} ||')
    if i < len(similarity)-1:
        print("-"*56)
print('='*56)

print(f'\nBest Match:- {products[best_match_index]}\n')