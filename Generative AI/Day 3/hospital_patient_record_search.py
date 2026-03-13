# Scenario: Hospital Patient Record Search
# - Problem: Doctors and nurses often type queries like “heart attack” or “chest pain emergency”, but the hospital’s electronic health record (EHR) system stores diagnoses as “acute myocardial infarction”. Traditional keyword search fails because medical staff use everyday language while records use clinical terminology.
# - Solution: Use embeddings to capture semantic meaning. Embeddings understand that “heart attack” and “acute myocardial infarction” are related concepts. This allows the search system to return the right patient records even when the query wording doesn’t match the stored terminology.

from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Example patient records stored in hospital system
diagnosis_texts = [
    "acute myocardial infarction",
    "type 2 diabetes mellitus",
    "chronic obstructive pulmonary disease",
    "hypertension",
    "asthma"
]

# Convert diagnoses to embeddings
record_embeddings = model.encode(diagnosis_texts)

# Doctor search query
query = "heart attack emergency"

# Convert query to embedding
query_embedding = model.encode([query])

# Compute cosine similarity
similarities = model.similarity(query_embedding, record_embeddings)
print(f"Similarity: {similarities}")

# Get most similar record
best_match_index = np.argmax(similarities)

# Print results
print("Doctor Query:", query)
print("\nMost Relevant Record:")
print(diagnosis_texts[best_match_index])

print("\nSimilarity Scores:")
for i, score in enumerate(similarities[0]):
    print(f"{diagnosis_texts[i]} -> {score:.3f}") 