import streamlit as st

#Langchain imports for LLM
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


st.title("ChatBot for Medical Support with RAG")


#Setup session state variable to hold the chat history

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

#Display chat history
for message in st.session_state.chat_history:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input("Enter your message here")

if(prompt):
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({'role':'user', 'content':prompt})


    #Initialize the LLM model

    model = "llama-3.3-70b-versatile"
    groq_chat = ChatGroq(
        groq_api_key = os.environ.get("GROQ_API_KEY"),
        model_name=model
    )

    #Create a prompt template
    groq_sys_prompt = ChatPromptTemplate.from_template("""
    You are a medical professional of many years of experience providing support to a patient with a medical condition. You need to give the most accurate 
                                                       solution for the patient's problem. 
                                                       Now patient is asking the following question: {user_prompt}
                                                       Start giving answer and be poisitive and give assurance everything will be alright"""
    )

    chain = groq_sys_prompt | groq_chat | StrOutputParser()

    response = chain.invoke({"user_prompt": prompt})
    # response = "Hi there! I am a FuDDu chatbot."
    st.chat_message("bot").markdown(response)
    st.session_state.chat_history.append({'role': 'bot', 'content': response})


