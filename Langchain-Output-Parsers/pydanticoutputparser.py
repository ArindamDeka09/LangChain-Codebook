from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

print(f"API Key found: {os.getenv('GOOGLE_API_KEY')[:5]}****")
# Setup the Chat Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

class Person(BaseModel):
    
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    city: str = Field(description="Name of the city where the person lives")


parser = PydanticOutputParser(pydantic_object=Person)


template = PromptTemplate(
    template="generate a random person with name, age and city of a fictional {place} person \n {format_instructions}",
    input_variables=['place'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)


# prompt = template.format(place="Indian")

# print(prompt)

# result = model.invoke(prompt)


# final_result = parser.parse(result.content)

#USING CHAIN 

chain = template | model | parser

final_result = chain.invoke({'place': 'Indian'})


print(final_result)

