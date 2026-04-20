from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel 
import os


load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)


prompt1 = PromptTemplate(
    template = "Generate a LinkedIn post about {topic}",
    input_variables = ['topic'] 
)

prompt2 = PromptTemplate(
    template = "Generate a tweet about {topic}",
    input_variables = ['topic'] 
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'LinkedIn_post': RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic': 'AI in healthcare'}) 

print(result['tweet'])
print(result['LinkedIn_post'])