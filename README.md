# News article research Tool

This is a Streamlit-based application designed to allow users to input URLs of news articles, process them to create embeddings, and store them in a FAISS vector store. Users can then ask questions related to the content of the articles, and the tool retrieves relevant information using a Hugging Face model for question answering.

## File Structure

news-research-tool/ 
├── app.py 
├── requirements.txt 
├── README.md 
└── utils/ 
├── data_loader.py
├── text_splitter.py 
├── embeddings.py 
├── vectorstore.py 
└── qa.py

## Project Description

### Overview

This tool leverages multiple technologies to create an efficient news research tool. 

The main components of the project include:

- **Streamlit**: A fast way to build and share data apps.
- **LangChain**: A framework for developing applications powered by language models.
- **Hugging Face Transformers**: A library for state-of-the-art natural language processing.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors.

### Technologies Used

1. **Streamlit**: Used to build the interactive web interface.
2. **LangChain**: Utilized for its text splitting and vector storage capabilities.
3. **Hugging Face**: Provides the pre-trained models for embeddings and question answering.
4. **FAISS**: Used to store and retrieve document embeddings efficiently.

### Key Components

- **Data Loading**: Loading data from unstructured URLs.
- **Text Splitting**: Splitting text into manageable chunks.
- **Embeddings**: Creating vector representations of the text.
- **Vector Storage**: Storing and retrieving these vectors using FAISS.
- **Question Answering**: Using a pre-trained Hugging Face model to answer questions based on the retrieved documents.

## Installation

### Prerequisites

- Python 3.6 or higher
- `pip` package manager

Usage
1.	Input URLs:
o	Enter up to 3 URLs of news articles in the sidebar.
2.	Process URLs:
o	Click on "Process URLs" to load and process the articles.
o	The tool will display messages indicating the progress of data loading, text splitting, and embedding vector creation.
3.	Ask Questions:
o	Enter a question related to the content of the articles in the text input field.
o	The tool will retrieve relevant information and display the answer along with the sources.

Detailed Steps

1. Data Loading
The UnstructuredURLLoader from langchain.document_loaders is used to load data from the provided URLs. It reads the content from the URLs and loads them into the application.
2. Text Splitting
The RecursiveCharacterTextSplitter from langchain.text_splitter is used to split the loaded text data into smaller chunks that can fit into the model's context window. This is crucial for handling large documents efficiently.
3. Creating Embeddings
The HuggingFaceEmbeddings from langchain_huggingface is used to create embeddings from the document texts. These embeddings capture the semantic meaning of the text, which allows for efficient similarity search.
4. Saving the FAISS Index
The FAISS index, which stores the embeddings, is saved to a pickle file for later retrieval. This allows the application to quickly load the precomputed embeddings without recalculating them.
5. Loading the Model and Tokenizer
The tokenizer and model for the question-answering pipeline are loaded from Hugging Face. This pipeline is responsible for processing the input question and generating an answer.
6. Creating the QA Pipeline
A HuggingFacePipeline is created with the QA model to handle the question answering.
7. Retrieving Relevant Documents
The vector store is used to retrieve documents relevant to the query. This is done using a retriever object, which searches the FAISS index for the most similar embeddings to the query.
8. Formatting the QA Input
The context for the QA model is formed by concatenating the content of the retrieved documents. This context is then passed along with the query to the QA model.
9. Getting the Answer
The QA model processes the input to generate an answer. The result contains the answer and possibly other metadata.
10. Displaying the Answer and Sources
The answer and sources are displayed in the Streamlit interface. This includes displaying the answer to the user's question and the sources of the information.

**Contributing**
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.




![Screenshot_19-6-2024_184344_localhost](https://github.com/manpreet171/News-Research-Bot/assets/172519023/f2710d35-0567-4d4f-8445-464c5bb880fd)




