from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import time
import os

load_dotenv()

print(f"API Key found: {os.getenv('GOOGLE_API_KEY')[:5]}****")
# Setup the Chat Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Use the stable 2026 identifier
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser # Langchain expression language

result = chain.invoke({'topic': 'Unemployment in India'})

print(result)  


