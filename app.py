import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_community.vectorstores import Cassandra
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
import cassio

# Load environment variables
load_dotenv()
token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
database_id = os.getenv("ASTRA_DB_ID")

def extract_pdf_text(pdf_docs):
    raw_text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content
    return raw_text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    text_chunks = text_splitter.split_text(raw_text)
    return text_chunks

def get_vector_store(text_chunks):
    cassio.init(token=token, database_id=database_id)
    embedding = OpenAIEmbeddings()
    astra_vector_store = Cassandra(
        embedding=embedding,
        table_name="qa_mini_demo",
    )
    astra_vector_store.add_texts(text_chunks)
    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
    return astra_vector_store, astra_vector_index

def get_conversation_chain(vstore):
    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    philo_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    provided context just say, "answer is not available in the context". Don't provide the wrong answer
    and don't include anything in your answer that is not provided in the context. 
    Do not come up with answers of your own.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    philo_prompt = ChatPromptTemplate.from_template(philo_template)
    llm = ChatOpenAI()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | philo_prompt
        | llm
        | StrOutputParser()
    )
    return chain

def main():
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.header("Chat with PDF")

    # Custom CSS to adjust the width of the input field
    st.markdown(
        """
        <style>
        .stTextInput>div>div>input {
            width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if 'astra_vector_index' not in st.session_state:
        st.session_state.astra_vector_index = None
    if 'chain' not in st.session_state:
        st.session_state.chain = None

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing..."):
                raw_text = extract_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vstore, st.session_state.astra_vector_index = get_vector_store(text_chunks)
                st.session_state.chain = get_conversation_chain(vstore)
                st.success("Done")

    user_question = st.text_input("Ask a question from the PDF file")

    if user_question:
        if st.session_state.astra_vector_index is None:
            st.error("Please upload and process a PDF file first.")
        else:
            answer = st.session_state.chain.invoke(user_question).strip()
            st.write("Answer:", answer)

if __name__ == "__main__":
    main()