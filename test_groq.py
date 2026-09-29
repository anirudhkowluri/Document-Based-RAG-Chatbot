import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

print("Key exists:", bool(groq_api_key))

if groq_api_key:
    print("Key prefix:", groq_api_key[:8])

llm = ChatGroq(
    api_key=groq_api_key,
    model="openai/gpt-oss-120b",
    temperature=0.2
)

response = llm.invoke("Say hello in one sentence.")

print(response.content)