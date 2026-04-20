import os
from dotenv import load_dotenv
# 1. Use ChatGoogleGenerativeAI (the modern class)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# 2. Specify the 2026 model name and your API key
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# 3. Gemini requires this specific list format
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Tell me about Langchain")
]

# 4. Get the response
result = model.invoke(messages)

# 5. Append the AI's response to the list to keep the conversation going
messages.append(AIMessage(content=result.content))

# 6. Print the result
print("AI Response:", result.content)