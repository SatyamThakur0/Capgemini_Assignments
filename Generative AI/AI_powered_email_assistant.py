from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

print(generator("Artificial Intelligence will", max_length=30))

# Scenario: Research Assistant for Academic Papers
# Imagine you’re building a tool for university researchers who analyze large collections of academic papers.
# - A researcher uploads a sentence:
# “Deep learning models are powerful”
# - Your system uses the BERT tokenizer to break the sentence into smaller units (tokens).

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Deep learning models are powerful"

tokens = tokenizer.tokenize(text)

print(tokens)


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


#  Scenario: Scriptwriting Assistant for Film Production
# A film studio is experimenting with AI to help writers brainstorm dialogue and plot ideas.
# - The writer provides the prompt:
# “The future of Artificial Intelligence”
# - The system generates two versions of the continuation:
# - Temperature = 0.2 (low randomness)
# Output is more predictable and formal, e.g.:
# “The future of Artificial Intelligence will improve industries, enhance productivity, and reshape education.”
# → This is useful for serious, factual narration in a documentary script.
# - Temperature = 0.8 (higher randomness)
# Output is more imaginative and varied, e.g.:
# “The future of Artificial Intelligence dances with uncertainty, weaving stories of machines dreaming and societies reborn.”
# → This is perfect for creative dialogue or speculative sci‑fi storytelling

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = input("Enter a prompt: ")

output = generator(prompt, max_length=50, temperature=0.7)

print(output[0]["generated_text"])