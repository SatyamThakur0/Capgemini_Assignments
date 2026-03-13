from transformers import pipeline

# Load GPT-2 text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Prompt from writer
prompt = input("Enter a prompt: ")

# Low temperature (more predictable)
output_low = generator(
    prompt,
    max_length=50,
    temperature=0.2,
    num_return_sequences=1
)

# High temperature (more creative)
output_high = generator(
    prompt,
    max_length=50,
    temperature=0.8,
    num_return_sequences=1
)

print("\nLow Temperature (0.2) Output:")
print(output_low[0]["generated_text"])
print('*'*30)

print("\nHigh Temperature (0.8) Output:")
print(output_high[0]["generated_text"])