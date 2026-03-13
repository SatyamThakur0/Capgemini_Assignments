# 🎓 Scenario: Student Brainstorming Tool for Essay Writing
# A college student is struggling to start their essay.
# - The system prompts them:
# “Enter a prompt:”
# - The student types:
# “The impact of technology on education”
# - The GPT‑2 model generates a continuation up to 50 tokens, for example:
# “The impact of technology on education has transformed classrooms, enabling online learning,
# personalized study plans, and global collaboration among students.”
from transformers import pipeline

# Load GPT-2 model for text generation
generator = pipeline("text-generation", model="gpt2")

# Ask student for essay topic
prompt = input("Enter a prompt: ")

# Generate continuation (up to 50 tokens)
output = generator(prompt, max_length=50)

# Print generated essay idea
print("\nGenerated Idea:")
print(output[0]["generated_text"])