from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda 
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal 

load_dotenv()

print(f"API Key found: {os.getenv('GOOGLE_API_KEY')[:5]}****")

model1 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY"), 
    temperature=0
)

parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description ="Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = "Classify the sentiment of the following text as positive, negative, or neutral \n {feedback} \n {format_instructions}",
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template = "Write an appropriate response to the positive feedback \n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template = "Write an appropriate response to the negative feedback \n {feedback}",      
    input_variables=['feedback']
)

classifier_chain = prompt1 | model1 | parser2

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model1 | parser), 
    (lambda x: x.sentiment == 'negative', prompt3 | model1 | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)


chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': "I love the new design of your website! It's so user-friendly and visually appealing."}))

chain.get_graph().print_ascii()