import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
# Use ChatGoogleGenerativeAI instead of GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
print(f"API Key found: {os.getenv('GOOGLE_API_KEY')[:5]}****") # Only shows first 5 chars for safety

# Setup the Chat Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# Your schema remains the same
class Review(TypedDict):

    key_themes: Annotated[list[str], "Write down all the key themes mentioned in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "Return sentiment of the review, either 'positive', 'negative', or 'neutral'"]
    pros: Annotated[Optional[list[str]], "List the pros mentioned in the review inside a list. If no pros are mentioned, return an empty list"]
    cons: Annotated[Optional[list[str]], "List the cons mentioned in the review inside a list. If no cons are mentioned, return an empty list"]

# Apply structured output
structured_llm = model.with_structured_output(Review)

# Run the chain
result = structured_llm.invoke("The hardware is great, but the software feels bloated. There are too many pre-installed apps that I never use, and it slows down the device. Overall, I'm disappointed with the software experience. However, the hardware is top-notch and performs well for gaming and multitasking. Pros: Excellent build quality, great performance. Cons: Bloated software, too many pre-installed apps, slow software experience. Cons: Bloated software, too many pre-installed apps, slow software experience. The best part is the hardware, but the software really drags it down. I wish they would optimize the software and remove unnecessary apps to improve the overall user experience, while the worst part is the software, which feels bloated and slows down the device. I hope they can improve the software experience in future updates.")

print(result)
print(result['key_themes'])
print(result['summary'])
print(result['sentiment'])
print(result['pros'])
print(result['cons'])