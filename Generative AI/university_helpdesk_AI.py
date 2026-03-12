# Scenario-Based Assessment
# Case Study: AI Assistant for a University Helpdesk

# A large university receives thousands of student queries daily related to admissions, fees, hostel facilities, course
#  schedules, and exam results.

# Currently, these queries are answered manually by staff, causing:
# Slow response time
# High operational cost
# Inconsistent responses

# The university plans to develop an AI-powered chatbot using Generative AI and Large Language Models (LLMs) to
#  automatically respond to student questions.

# You are part of the AI consulting team tasked with designing the solution.

# ==========================================
# UNIVERSITY AI HELPDESK CHATBOT
# Demonstrates:
# Generative AI
# Large Language Models (LLMs)
# Tokenization
# Transformer Inference
# Prompt Engineering
# ========================================== 

# Install required libraries first
# pip install transformers torch


# ==========================================
# UNIVERSITY AI HELPDESK CHATBOT
# Using Hugging Face Inference API
# No local model download
# ==========================================

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient()

print("="*40, "UNIVERSITY AI HELPDESK", "="*40)

while True:

    question = input("Student: ")

    if question.lower() in ["exit", "quit", "0"]:
        print("Chatbot: Thank you. Have a great day!")
        break

    prompt = f"""
You are an AI assistant for a university helpdesk.
Answer student queries politely and clearly.

Student Question: {question}

Answer:
"""

    response = client.text_generation(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        prompt=prompt,
        max_new_tokens=200
    )

    print("Chatbot:", response)