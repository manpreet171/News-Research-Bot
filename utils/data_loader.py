from langchain.document_loaders import UnstructuredURLLoader

def load_data(urls):
    loader = UnstructuredURLLoader(urls=urls)
    return loader.load()
