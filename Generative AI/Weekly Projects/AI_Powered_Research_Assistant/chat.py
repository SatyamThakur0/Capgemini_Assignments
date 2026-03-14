from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()


system_prompt = """
You are an AI research assistant.
Your job is to analyze academic text and generate concise summaries
that help students quickly understand research articles.
"""

human_prompt = """
Summarize the following research article in 5 clear bullet points.

Article:
{text}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt)
])

model = ChatMistralAI(
    model='mistral-medium-latest'
)
para = input("Give your Paragraph: ")
final_prompt = prompt.invoke(
    {
        'text': para
    }
)
res = model.invoke(final_prompt)
print('\n','='*130)
print('\n',res.content)
print('\n','='*130)