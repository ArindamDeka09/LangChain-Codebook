from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

print(f"API Key found: {os.getenv('GOOGLE_API_KEY')[:5]}****") # Only shows first 5 chars for safety    
# Setup the Chat Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)


# 1st prompt -> detailed report

template1 = PromptTemplate(
    template = "Write a detailed report about the following topic: {topic}",
    input_variables=["topic"]
)

# 2nd prompt -> summary

template2 = PromptTemplate(
    template = "Write a 5 line summary about the following topic:/n {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'Black Hole'})

print(result)
