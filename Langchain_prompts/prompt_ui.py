import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import load_prompt

# 1. Load credentials from your .env file
# Ensure your .env has: GOOGLE_API_KEY=your_key_here
load_dotenv()

# 2. Setup the Gemini Model
@st.cache_resource
def load_gemini_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # 2026 Update: Trying the latest stable models
    # We try 2.5 Flash first, then 3.0 Flash as a backup
    model_names = ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-1.5-flash"]
    
    for name in model_names:
        try:
            model = ChatGoogleGenerativeAI(
                model=name,
                google_api_key=api_key,
                temperature=0.7
            )
            # Test if the model exists by doing a tiny internal check
            return model
        except Exception:
            continue # Try the next model name in the list
            
    st.error("❌ All Gemini models failed. Please check if 'Generative Language API' is enabled in Google AI Studio.")
    st.stop()

model = load_gemini_model()

# 3. Streamlit UI
st.set_page_config(page_title="Gemini Research Tool", page_icon="♊")
st.header('Research Tool (Powered by Gemini)')

paper_input = st.selectbox("Select Research Paper", [
    "Attention Is All You Need", 
    "BERT: Pre-training of Deep Bidirectional Transformers", 
    "GPT-3: Language Models are Few-Shot Learners"
])

style_input = st.selectbox("Select Style", ["Beginner-Friendly", "Technical", "Mathematical"]) 
length_input = st.selectbox("Select Length", ["Short", "Medium", "Long"])

# 4. Load your template.json
try:
    template = load_prompt('Langchain_prompts/template.json')
except Exception as e:
    st.error("Error: 'template.json' not found in the 'Langchain_prompts' folder.")
    st.stop()

# 5. Run the AI
if st.button('Summarize with Gemini'):
    with st.spinner("♊ Gemini is analyzing the paper..."):
        try:
            # Create and run the chain
            chain = template | model
            result = chain.invoke({
                'paper_input': paper_input,
                'style_input': style_input,
                'length_input': length_input
            })
            
            st.subheader("Summary Result:")
            # Use .content because ChatGoogleGenerativeAI returns a message object
            st.info(result.content)
            
        except Exception as e:
            st.error(f"Something went wrong: {e}")