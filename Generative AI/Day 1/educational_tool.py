# Scenario: Educational Tool for Classroom Learning
# A high school teacher is introducing students to AI‑powered writing assistants.
# - The teacher wants to show how different models produce different styles of text.
# - They use the same prompt:
# “Artificial Intelligence will transform”
# - The system generates two outputs:
# - GPT‑2 Output
# More elaborate continuation, e.g.:
# “Artificial Intelligence will transform the way societies function, influencing education, healthcare, and the future of human creativity.”
# → Great for showing students how AI can generate rich, detailed ideas.
# - DistilGPT‑2 Output
# Shorter, faster continuation, e.g.:
# “Artificial Intelligence will transform our daily lives.”
# → Useful for demonstrating concise summaries.
from transformers import pipeline

# Load models
gpt2 = pipeline("text-generation", model="gpt2")
distilgpt2 = pipeline("text-generation", model="distilgpt2")

prompt = "Artificial Intelligence will transform"

# Generate text
print("GPT-2 Output:")
print(gpt2(prompt, max_length=30)[0]['generated_text'])

print("\nDistilGPT-2 Output:")
print(distilgpt2(prompt, max_length=30)[0]['generated_text'])