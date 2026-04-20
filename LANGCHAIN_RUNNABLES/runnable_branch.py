from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic'] 
)

prompt2 = PromptTemplate(
    template="Summarize the following report in 3 sentences: {text}",
    input_variables=['text']
)

parser = StrOutputParser()

# report_gen_chain = RunnableSequence(prompt1, model, parser)

report_gen_chain = prompt1 | model | parser  # runnable sequence is used very often so it can be written in this way as well and it is called LCEL - Langchain Expression Language

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic': 'the impact of US dollar in ruppees'}))