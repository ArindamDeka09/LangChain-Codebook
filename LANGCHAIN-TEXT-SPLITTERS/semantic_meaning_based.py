from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

text_splitter = SemanticChunker(
    embeddings, 
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

sample = """
The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials. It was built along the northern borders of China to protect against invasions and raids by nomadic groups from the Eurasian Steppe. The Indian Premier League (IPL) is a professional Twenty20 cricket league in India contested during April and May of every year by teams representing different cities. The league was founded by the Board of Control for Cricket in India (BCCI) in 2008, and is regarded as the world's most popular cricket league. The IPL has been credited with increasing the popularity of cricket in India and around the world, and has also been a major source of revenue for the BCCI and its players. The B.tech in AI is one of the most sought-after undergraduate programs in the field of artificial intelligence. It provides students with a strong foundation in computer science, mathematics, and AI concepts, preparing them for careers in AI research, development, and applications. The program typically includes courses on machine learning, deep learning, natural language processing, computer vision, and robotics, among others. Graduates of the B.tech in AI program are well-equipped to contribute to the rapidly evolving field of artificial intelligence and can pursue careers in various industries such as technology, healthcare, finance, and more. 

The history of surgery dates back to ancient times, with evidence of surgical procedures being performed as early as 3000 BCE. The Indian subcontinent has a rich history of surgical practices, with the ancient text Sushruta Samhita, written by the physician Sushruta, being one of the earliest known texts on surgery. The text describes various surgical techniques and procedures, including plastic surgery, cataract surgery, and even brain surgery. The practice of surgery has evolved significantly over the centuries, with advancements in medical knowledge, technology, and techniques leading to improved outcomes for patients. Today, surgery is a critical component of modern medicine, with a wide range of procedures being performed to treat various conditions and diseases. The history of surgery is a testament to human ingenuity and the desire to improve health and well-being, and it continues to evolve as new discoveries and innovations are made in the field of medicine.

"""


docs = text_splitter.split_text(sample)
print(len(docs))
print(docs)
print(docs[0])
print(docs[1])