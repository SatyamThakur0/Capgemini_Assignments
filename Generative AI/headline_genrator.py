# Scenario: AI‑Powered News Headline Generator
# A digital media company wants to help journalists brainstorm headlines and opening lines for articles.
# - The system asks the journalist to enter a prompt (e.g., “Enter a prompt: The future of Artificial Intelligence”).
# - The GPT‑2 model then generates a continuation of the text, up to 50 tokens.
# - Example output:
# “The future of Artificial Intelligence will reshape industries, redefine creativity, and challenge
#  our understanding of human potential.”

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = input("Enter a prompt: ")

output = generator(prompt, max_length=50)

print(output[0]["generated_text"])
