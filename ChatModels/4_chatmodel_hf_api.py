# import os
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
# )

# model = ChatHuggingFace(llm=llm)

# model.invoke("What is the capital of France?")

# result = model.invoke("What is the capital of France?")

# print(result.content)


import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage

load_dotenv()

# 1. Initialize the endpoint with the 'conversational' task
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational",  # <-- CRITICAL: Must match model's design
    max_new_tokens=512,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# 2. Use the Chat wrapper which handles the conversational formatting
chat_model = ChatHuggingFace(llm=llm)

# 3. Pass a list of messages (required for conversational tasks)
messages = [
    HumanMessage(content="What is the capital of France?")
]

print("Asking Qwen via Conversational API...")
result = chat_model.invoke(messages)

print("\nResponse:")
print(result.content)