from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

print(f"API Key found: {os.getenv('GOOGLE_API_KEY')[:5]}****")
# Setup the Chat Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

class TopicFacts(BaseModel):
    fact1: str = Field(description="Fact 1 about the topic")
    fact2: str = Field(description="Fact 2 about the topic")
    fact3: str = Field(description="Fact 3 about the topic")

parser = JsonOutputParser(pydantic_object=TopicFacts)

template = PromptTemplate(
    template='Give 3 facts about the {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic': 'cricket'})
print(result)