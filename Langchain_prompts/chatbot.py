from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage # <--- MUST ADD THIS
from dotenv import load_dotenv
import os

load_dotenv()

# Specify the model name clearly
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash") 

chat_history = [
    SystemMessage(content="You are a helpful assistant")
]

while True: 
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input)) # Keep track of the conversation history
    if user_input.lower() == 'exit': 
        break
    
    # Wrap the string in a LIST [] and a HumanMessage()
    try:
        result = model.invoke(chat_history)
        chat_history.append(AIMessage(content=result.content)) # Add the AI's response to the history for context in future interactions
        print("AI: ", result.content)
    except Exception as e:
        print(f"Error: {e}")

print(chat_history)