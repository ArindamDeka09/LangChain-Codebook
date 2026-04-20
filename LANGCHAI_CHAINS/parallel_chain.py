from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough # NEW & CORRECT

load_dotenv()

print(f"API Key found: {os.getenv('GOOGLE_API_KEY')[:5]}****")
# Setup the Chat Model
model1 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Use the stable 2026 identifier
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

model2 = ChatGoogleGenerativeAI(
    model ="gemini-2.5-flash-lite", # Use the stable 2026 identifier
    google_api_key=os.getenv("GOOGLE_API_KEY"),     
    temperature=0   
)


prompt1 = PromptTemplate(
    template="Generate a dshort and simple notes from the following text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Generate a 5 short question answers from the following text \n {text}",
    input_variables=['text']
)


prompt3 = PromptTemplate(
    template = "Merge the provided notes and quiz into a single document \n Notes: {notes} and Quiz: {quiz}",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Unemployment in India is a significant socio-economic issue that has persisted for decades. The country has a large and growing population, which has led to a high demand for jobs. However, the supply of jobs has not kept pace with the demand, resulting in a high unemployment rate. The unemployment rate in India varies across different states and regions, with some areas experiencing higher rates than others. The government has implemented various policies and programs to address unemployment, such as skill development initiatives and job creation schemes. Despite these efforts, unemployment remains a challenge, particularly among the youth and marginalized communities. The COVID-19 pandemic further exacerbated the situation, leading to job losses and economic instability. Addressing unemployment in India requires a multi-faceted approach that includes improving education and skill development, promoting entrepreneurship, and creating a conducive environment for businesses to thrive.
"""


result = chain.invoke({'text': text})

print(result) 

chain.get_graph().print_ascii()