from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('LANGCHAIN_DOCUMENT_LOADERS/NexuxOps_Synopsys.pdf')

# text = "This is a sample text that we will split into smaller chunks based on character count. The text splitter will help us manage large documents by breaking them down into more manageable pieces. We can specify the chunk size and the amount of overlap between chunks to ensure that we don't lose important context. This is especially useful when working with language models that have a maximum token limit. By using a character-based text splitter, we can ensure that our chunks are of a consistent size, which can help improve the performance of our language model. The text splitter will take care of the splitting process, allowing us to focus on analyzing and processing the text rather than worrying about how to divide it up. Overall, the character-based text splitter is a powerful tool for managing and processing large texts, making it easier to work with language models and extract valuable insights from our data."

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator= ' '
)


result = splitter.split_documents(docs)

print(result[0].page_content) 

print(result[1].page_content)  


