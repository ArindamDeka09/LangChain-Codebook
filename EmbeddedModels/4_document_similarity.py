import os
import numpy as np
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# 1. Setup local cache on your D: drive
os.environ['HF_HOME'] = 'D:/HF_cache'
load_dotenv()

# 2. Use the free Hugging Face model instead of OpenAI
# This runs locally on your Intel i5 CPU
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting style.",
    "Sachin Tendulkar is a legendary Indian cricketer often referred to as the 'God of Cricket'.",
    "M.S. Dhoni is a former Indian cricketer and captain of the Indian national team.",
    "Rohit Sharma is an Indian cricketer known for his ability to score big centuries.",
    "Jasprit Bumrah is an Indian cricketer known for his fast bowling and unique action."
]

query = "tell me about Rohit Sharma"

# 3. Generate embeddings (This converts text to 384-dimensional vectors)
query_embedding = embedding.embed_query(query)
doc_embeddings = embedding.embed_documents(documents)

# 4. Calculate similarity scores
# We wrap query_embedding in [] to make it a 2D array for sklearn
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# 5. Find the best match
index = np.argmax(scores)
best_score = scores[index]

print(f"Query: {query}")
print(f"Best Match (Score: {best_score:.4f}):")
print(documents[index])