# Scenario: Hospital FAQ Assistant
# A hospital wants to build a chatbot that helps patients quickly find answers to common questions.
# - Problem: Patients ask questions in different ways. For example:
# - “How do I book an appointment?”
# - “What’s the process to schedule a doctor visit?”
# - “Can I see a physician tomorrow?”
# Traditional keyword search might miss these connections.
# - Solution: Convert FAQs and patient queries into embeddings, then measure similarity to find the closest match.


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


query_embedding = np.array([0.2, 0.3, 0.8])

product_embeddings = {
    "How do I book an appointment": np.array([0.2, 0.7, 0.6]),
    "What's the process to schedule a doctor visit?": np.array([0.9, 0.1, 0.2]),
    "“Can I see a physician tomorrow?”": np.array([0.3, 0.4, 0.5])
}

for product, embedding in product_embeddings.items():
    similarity = np.dot(query_embedding, embedding)  # fast similarity check
    cosine_sim = cosine_similarity([query_embedding], [embedding])[0][0]  # normalized similarity
    print(f"{product} → Dot Product: {similarity:.3f}, Cosine Similarity: {cosine_sim:.3f}")


best_match = max(product_embeddings.items(), key=lambda x: np.dot(query_embedding, x[1]))
print("\nValid Question:", best_match[0])