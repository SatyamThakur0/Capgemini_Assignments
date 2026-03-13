#  Scenario: Game Storyline Generator
# You’re designing a fantasy role‑playing game and need quick ideas for quests and dialogue.
# - The developer types a prompt:
# “The hero enters the ancient forest and discovers…”
# - The GPT‑2 pipeline continues the story:
# “The hero enters the ancient forest and discovers a hidden village where the people whisper of
#  a cursed artifact buried beneath the old ruins.”
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

model = init_chat_model("mistral-medium-latest")


prompt=["You're designing a fantasy role playing and need quick ideas for quest and dialoge"]

while(True):
    question=input("Enter your line:")
    if question=='0':
        break
    prompt.append(question)
    res=model.invoke(prompt)
    print(res.content)