from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """

class Student:
    def __init__(self, name, age):
        self.name = name    
        self.age = age
    def study(self, subject):
        return f"{self.name} is studying {subject}."
    def introduce(self):
        return f"Hello, my name is {self.name} and I am {self.age} years old."
        

    if __name__ == "__main__":
        student1 = Student("Alice", 20)
        print(student1.introduce())
        print(student1.study("Mathematics"))
    
    else:
        print("This code is being imported as a module, not run directly.") 

"""


# Initializing the text splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, 
    chunk_size=300, 
    chunk_overlap=0
)


# Perform the split
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])