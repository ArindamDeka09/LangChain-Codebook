from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser 

load_dotenv()

print(f"API Key found: {os.getenv('GOOGLE_API_KEY')[:5]}****") # Only shows first 5 chars for safety    
# Setup the Chat Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

parser = JsonOutputParser()

template = PromptTemplate(
    template = "Give me the name, age and city of a fictional person \n {format_instructions}",
    input_variables=[],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# prompt = template.format()

# print(prompt)

# result = model.invoke(prompt)
 
# final_result = parser.parse(result.content)

# print(final_result)
# print(type(final_result))


#OR

chain = template | model | parser

result = chain.invoke({})

print(result)
print(type(result))