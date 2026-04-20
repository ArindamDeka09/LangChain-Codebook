from langchain_community.document_loaders import WebBaseLoader
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
    template="Answer the following questions \n {question} from the following text - \n {text}",
    input_variables=['text', 'question']    
)

parser = StrOutputParser()

url = "https://www.amazon.in/ASUS-Processor-4050-6GB-140WTGP-FX677VU-RL055WS/dp/B0DXF2M5S5/?_encoding=UTF8&pd_rd_w=Co5Yp&content-id=amzn1.sym.7998c9a0-d63d-4775-a557-86e41b560555&pf_rd_p=7998c9a0-d63d-4775-a557-86e41b560555&pf_rd_r=F70EG2V9P8QAMD32XGGK&pd_rd_wg=ABo2x&pd_rd_r=23fcc362-bd77-406c-8f37-11e928b68570&ref_=pd_hp_d_atf_dealz_cs&th=1"
loader = WebBaseLoader(url)

docs = loader.load()



chain = prompt | model | parser

print(chain.invoke({'question': 'What is the RAM and storage capacity of the laptop?', 'text': docs[0].page_content})) 