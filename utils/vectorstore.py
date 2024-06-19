# utils/vectorstore.py

import pickle
import faiss
import streamlit as st

def save_vectorstore(vectorstore, file_path):
    try:
        with open(file_path, "wb") as f:
            pickle.dump({"index": faiss.serialize_index(vectorstore["index"]),
                         "metadata": vectorstore["metadata"]}, f)
    except Exception as e:
        st.error(f"Error saving vectorstore: {e}")

def load_vectorstore(file_path):
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            index = faiss.deserialize_index(data["index"])
            metadata = data["metadata"]
            return {"index": index, "metadata": metadata}
    except Exception as e:
        st.error(f"Error loading vectorstore: {e}")
        return None

def retrieve_chunks(vectorstore, query_embedding, top_k=5):
    try:
        distances, indices = vectorstore["index"].search(query_embedding, top_k)
        retrieved_docs = [vectorstore["metadata"][idx] for idx in indices[0]]
        return retrieved_docs
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return []
