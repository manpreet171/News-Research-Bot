import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, pipeline

load_dotenv()  # take environment variables from .env

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"

main_placeholder = st.empty()

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...")
    data = loader.load()
    
    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=150
    )
    main_placeholder.text("Text Splitter...Started...")
    docs = text_splitter.split_documents(data)
    
    # Create embeddings using HuggingFaceEmbeddings
    model_path = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(model_name=model_path, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    
    texts = [doc.page_content for doc in docs]
    metadatas = [{"source": doc.metadata["source"]} for doc in docs]
    
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    main_placeholder.text("Embedding Vector Started Building...")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            
            # Load the tokenizer and model for the question-answering pipeline
            model_name = "Intel/dynamic_tinybert"
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)
            question_answerer = pipeline("question-answering", model=model_name, tokenizer=tokenizer, return_tensors='pt')
            
            # Create a HuggingFacePipeline with the QA model
            llm = HuggingFacePipeline(pipeline=question_answerer, model_kwargs={"temperature": 0.7, "max_length": 512})
            
            # Create a retriever and QA chain
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            retrieved_docs = retriever.get_relevant_documents(query)
            
            # Format the context for the QA pipeline
            context = " ".join([doc.page_content for doc in retrieved_docs])
            qa_input = {"question": query, "context": context}
            
            result = question_answerer(qa_input)
            
            st.header("Answer")
            st.write(result['answer'])

            st.subheader("Sources:")
            for doc in retrieved_docs:
                st.write(doc.metadata.get("source", "Unknown source"))
