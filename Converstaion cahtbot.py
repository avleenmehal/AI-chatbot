import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

st.title("ChatBot for Medical Support with RAG")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize memory with proper message schema
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"
    )

# Display chat history
for message in st.session_state.chat_history:
    st.chat_message(message.type.replace("human", "user")).markdown(message.content)

@st.cache_resource
def get_vector_store():
    # Path to your CSV
    csv_file_path = "./train_data_chatbot.csv"
    

    loader = CSVLoader(
        file_path=csv_file_path,
        encoding='utf-8'  # or your file's encoding if different
    )
    
    # 2. Pass the loader to VectorstoreIndexCreator
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders([loader])
    
    return index.vectorstore
# Custom prompt with explicit chat history handling
custom_prompt = ChatPromptTemplate.from_template(
    """You are an AI medical researcher. Use the following context and conversation history to answer the question.
    Maintain a friendly tone and include appropriate humor when suitable.
    Below are some examples on how you will be interacting with the user:
    Example 1: User says I have cold.
    AI bot: Sorry to hear but its good you'll have some rest days. Also you can take Sumo cold tablets one a day along with Paracetamol to make you feel better.

    Example 2: User says I have Cancer
    Ai Bot: Oh my god, jokes apart , contact doctor as soon as possible and you'll be fine. But hey you'll have your dream sportstar visit you in person in the hospital.
    
    Context: {context}
    
    Conversation History:
    {chat_history}
    
    Question: {question}
    
    Helpful Answer:"""
)

def format_chat_history(history):
    formatted = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")
    return "\n".join(formatted)

prompt = st.chat_input("Enter your message here")

if prompt:
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    try:
        vectorstore = get_vector_store()
        groq_chat = ChatGroq(
            groq_api_key="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",
            model_name="llama-3.3-70b-versatile"
        )

        # Create conversation chain with proper history handling
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=groq_chat,
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            memory=st.session_state.memory,
            chain_type="stuff",
            combine_docs_chain_kwargs={
                "prompt": custom_prompt,
                "document_prompt": ChatPromptTemplate.from_template("{page_content}")
            },
            get_chat_history=lambda h: format_chat_history(h),
            verbose=True,
            return_source_documents=True
        )

        # Get response with formatted history
        result = qa_chain({
            "question": prompt,
            "chat_history": format_chat_history(st.session_state.chat_history)
        })
        response = result["answer"]

        # Update chat history
        st.session_state.chat_history.append(AIMessage(content=response))
        st.session_state.memory.save_context(
            {"question": prompt},
            {"answer": response}
        )

        # Display bot response
        st.chat_message("bot").markdown(response)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")