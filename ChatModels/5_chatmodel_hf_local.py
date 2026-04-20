# # # from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# # # import os

# # # os.environ['HF_HOME'] = 'D:/HF_cache'  # Set this to your desired cache directory
# # # llm = HuggingFacePipeline.from_model_id(
# # #     model_id="Qwen/Qwen2.5-7B-Instruct",
# # #     task="conversational",
# # #     pipeline_kwargs={"max_new_tokens": 512, "temperature": 0.7},
# # # )
# # # model = ChatHuggingFace(llm=llm)

# # # result = model.invoke("What is the capital of France?")

# # # print(result.content)



# # import os
# # from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# # from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# # # Set your cache directory
# # os.environ['HF_HOME'] = 'D:/HF_cache'

# # model_id = "Qwen/Qwen2.5-7B-Instruct"

# # # 1. Load the model using the correct Causal class
# # model = AutoModelForCausalLM.from_pretrained(
# #     model_id, 
# #     device_map="auto", # Automatically uses GPU if available
# #     torch_dtype="auto"
# # )
# # tokenizer = AutoTokenizer.from_pretrained(model_id)

# # # 2. Create the pipeline with the correct task
# # pipe = pipeline(
# #     "text-generation", # Use this instead of 'conversational' for local
# #     model=model,
# #     tokenizer=tokenizer,
# #     max_new_tokens=512,
# #     temperature=0.7
# # )

# # # 3. Wrap it in LangChain
# # llm = HuggingFacePipeline(pipeline=pipe)
# # chat_model = ChatHuggingFace(llm=llm)

# # # 4. Invoke
# # result = chat_model.invoke("What is the capital of France?")
# # print(result.content)


# import os
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# # 1. Set your cache to the D: drive as before
# os.environ['HF_HOME'] = 'D:/HF_cache'

# # 2. Use the 0.5B model (approx 1GB download)
# model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# print(f"Loading {model_id}...")

# # 3. Load model and tokenizer
# # We use device_map="auto" to use your GPU if you have one, or CPU otherwise
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, 
#     device_map="auto", 
#     torch_dtype="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# # 4. Create the local pipeline
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     temperature=0.7,
#     do_sample=True
# )

# # 5. Wrap in LangChain
# llm = HuggingFacePipeline(pipeline=pipe)
# chat_model = ChatHuggingFace(llm=llm)

# # 6. Test the model
# print("\n--- Asking Local AI ---")
# result = chat_model.invoke("What is the capital of France?")
# print(f"Response: {result.content}")



import os
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Using your D: drive to keep the C: drive (OS) fast
os.environ['HF_HOME'] = 'D:/HF_cache'

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# 'cpu' is used here because the Slim 5i usually lacks a dedicated NVIDIA GPU
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="cpu", 
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256, # Shorter limit for faster CPU responses
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=pipe)
chat_model = ChatHuggingFace(llm=llm)

print("Local AI is ready on your IdeaPad!")
response = chat_model.invoke("Give me a one-sentence tip for a BCA student.")
print(f"\nAI: {response.content}")  