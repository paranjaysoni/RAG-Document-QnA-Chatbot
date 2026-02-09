import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key, model="llama-3.3-70b-versatile")

prompt=ChatPromptTemplate.from_template(
    '''
    Answer the question based on provided context only.
    Please provide the most accurate response based on the question.

    <context>
    {context}
    </context>
    Question{input}
    '''
)

# Using this our application will have some memory to refer to:
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings() #Embeddings
        st.session_state.loader=PyPDFDirectoryLoader("/research_papers") #Data Ingestion
        st.session_state.docs=st.session_state.loader.load() #Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

user_prompt=st.text_input("Enter Your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Databaseis ready")

if user_prompt:
    document_chain=create_stuff_documents_chain(llm, prompt)
    retriever=st.session_state.vectors.as_retriver()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    response=retrieval_chain.invoke({'input': user_prompt})
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('----------------------')
