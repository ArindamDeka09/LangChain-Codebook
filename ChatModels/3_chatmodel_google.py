import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# We initialize the model
# Since you're interested in AI Agents, we use gemini-1.5-pro
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print("Asking Gemini...")
# This is the line that actually sends the request to Google
result = llm.invoke("What is the capital of France?")

print("\nResponse from Gemini:")
print(result.content)