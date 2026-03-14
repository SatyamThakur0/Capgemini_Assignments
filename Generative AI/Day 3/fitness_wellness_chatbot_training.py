# Scenario: Fitness & Wellness Chatbot Training
# You feed it these sentences:
# - “Push-ups strengthen chest and triceps”
# - “Yoga improves flexibility and balance”
# - “Apples are rich in fiber”
# - “Cardio exercises boost heart health”
# 👉 In this case, the chatbot learns to distinguish 
# between exercise facts, nutrition facts, and general 
# wellness statements, so it can tailor its responses — 
# e.g., suggesting workouts, offering diet tips, or explaining health benefits

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

excercises = [
    "Push-ups strengthen chest and triceps",
    "Yoga improves flexibility and balance",
    "Apples are rich in fiber",
    "Cardio exercises boost heart health"
]

query = "suggesting workouts"

embeddings = model.encode(excercises)
query_embeddings = model.encode(query)

similarities = model.similarity(query_embeddings, embeddings)

max_query_index = np.argmax(similarities)

# print(embeddings)
print(f'\nHighest Similarity:- {excercises[max_query_index]}\n')

print('='*56)
for i, score in enumerate(similarities[0]):
    print(f'|| {excercises[i]:<40} --> {score:.3f} ||')
    if i < len(similarities[0])-1:
        print("-"*56)
print('='*56)