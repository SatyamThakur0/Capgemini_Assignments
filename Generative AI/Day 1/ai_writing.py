# Scenario: Exploring Creativity in AI Writing
# A university professor is running a workshop on creative writing with AI.
# - The professor gives students the prompt:
# “The future of Artificial Intelligence”
# - The system uses GPT‑2 to generate continuations of the sentence.
# To demonstrate how temperature affects creativity:
# - With temperature = 0.2 (low randomness), the output is more predictable and conservative, e.g.:
# “The future of Artificial Intelligence will bring advancements in healthcare, education, and business.”

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "The future of Artificial Intelligence"

print(generator(prompt, max_length=40, temperature=0.2))
print(generator(prompt, max_length=40, temperature=0.8))

