from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_data(data):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=150
    )
    return text_splitter.split_documents(data)
