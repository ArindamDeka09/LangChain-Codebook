from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='LANGCHAIN_DOCUMENT_LOADERS/Social_Network_Ads.csv')

data = loader.load()

print(len(data))
print(data[1])