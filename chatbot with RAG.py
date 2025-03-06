import streamlit as st

#Langchain imports for LLM
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

#Imports for RAG
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
# from langchain.chains import retrieval_qa

st.title("ChatBot for Medical Support with RAG")


#Setup session state variable to hold the chat history

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

#Display chat history
for message in st.session_state.chat_history:
    st.chat_message(message['role']).markdown(message['content'])

@st.cache_resource
def get_vector_store():
    pdf_name = "./attentionIsAllYouNeed.pdf"
    loaders = [PyPDFLoader(pdf_name)]
    # Createing chunks or vectors - we use ChromaDB
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)
    return index.vectorstore


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
    You are a AI researcher of many years of experience providing support to a student . You need to give the most accurate 
                                                       solution for the student's problem. 
                                                       Now student is asking the following question: {user_prompt}
                                                       Start giving answer and be a bit funny"""
    )

    try:
        vectorstore = get_vector_store()
        if vectorstore is None:
            st.error("Failed to create vector store")

        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type='stuff',
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True)

        result = chain({"query": prompt})
        response = result["result"]            

        # response = "Hi there! I am a FuDDu chatbot."
        st.chat_message("bot").markdown(response)
        st.session_state.chat_history.append({'role': 'bot', 'content': response})
    
    except Exception as e:
        st.error(f"An error occurred: [{str(e)}]")


