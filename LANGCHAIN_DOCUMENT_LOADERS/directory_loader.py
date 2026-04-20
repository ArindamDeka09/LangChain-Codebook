import os
print(os.listdir('LANGCHAIN_DOCUMENT_LOADERS/Multiple_Files'))
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader

loader = DirectoryLoader(
    path='LANGCHAIN_DOCUMENT_LOADERS/Multiple_Files',
    glob='*.docx',  # Change to docx
    loader_cls=Docx2txtLoader  # Use the Word loader
)

docs = loader.load()

print(docs)
print(len(docs))

# if docs:
#     print(type(docs[0].page_content))
#     print(docs[0].metadata)


# for document in docs:
#     print(document.metadata)

# Lazy loading example
docs_lazy = loader.lazy_load()

for doc in docs_lazy:
    print(doc.metadata) 