import os
import openai 
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

## LangSmith Tracking ##
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_PROJECT"] = "Q&A ChatBot with OpenAI"


prompt = ChatPromptTemplate.from_messages(
    [
        ('system','you are a helpful assistant please response to the user query'),
        ('user','Question : {question}')
    ]
)


def generate_response(question, api_key,llm,temperature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model=llm,temperature=temperature)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question':question})
    return answer

## Streamlit App
## Title of the app

st.title('Enhanced Q&A Chatbot with Open AI')

## Side Bar

st.sidebar.title('Settings')
api_key = st.sidebar.text_input('Enter your Open AI API key:',type='password')

## Drop down to select Open AI models
llm = st.sidebar.selectbox('Select an Open AI model', ['gpt-4o', 'gpt-4-turbo', 'gpt-4'])

## Adjust response parameters

temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## Main interface for user input

st.write('Go ahead and ask any question')
user_input = st.text_input('You:')

if user_input:
    response = generate_response(question=user_input,api_key=api_key,llm=llm, temperature=temperature,max_tokens=max_tokens)
    st.write(response)
else:
    st.write('Please provide the query')
