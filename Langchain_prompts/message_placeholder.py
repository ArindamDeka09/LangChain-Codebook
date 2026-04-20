from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

# chat template
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'), 
    ('human','{query}')
])

chat_history = []
#load chat history
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, 'chat_history.txt')

# USE THE VARIABLE HERE:
with open(file_path) as f:
    chat_history.extend(f.readlines())

print(chat_history)

# create prompt

prompt = chat_template.invoke({'chat_history': chat_history, 'query': HumanMessage(content='Where is my refund')})


print(prompt)