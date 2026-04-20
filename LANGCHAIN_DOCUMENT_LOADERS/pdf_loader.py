from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('NexuxOps_Synopsys.pdf')

docs = loader.load()

print(docs)

print(len(docs))

print(type(docs[0].page_content))
print(docs[1].metadata)

