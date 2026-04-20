from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

prompt = PromptTemplate(
    template="Write a summary of the following text: {text}",
    input_variables=['text']    
)

parser = StrOutputParser()


loader = TextLoader('LANGCHAIN_DOCUMENT_LOADERS/cricket.text', encoding='utf-8')   

docs = loader.load()


print(type(docs)) 


print(type(docs[0]))

print(docs[0].page_content)

print(docs[0].metadata)

chain = prompt | model | parser

chain.invoke({'text': docs[0].page_content})