from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence 
import os

from typer import prompt

load_dotenv()

# Setup the Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", # Stable 2026 identifier
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# Setup Prompt
prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain the concept of {concept} in simple terms.",
    input_variables=['concept']
)

parser = StrOutputParser()

# Define the Chain
chain = RunnableSequence(prompt1, model, parser, parser, prompt2, model, parser)

# Invoke AND Print (Crucial!)
result = chain.invoke({'topic': 'programming'})
print(result)