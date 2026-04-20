from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "Delhi is the capital of India."

documents = [
    "Delhi is the capital of India.",
    "Kolkata is the capital of West Bengal.",
    "Guwahati is the capital of Assam."
]

vectors = embedding.embed_documents(documents)

print("Vector for the query text:")
print(str(vectors[0]))