# import streamlit as st
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# import os
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS


# def get_pdf_text(get_docs):
#     text = ""
#     for file in get_docs:
#         file_extension = os.path.splitext(file.name)[1].lower()
        
#         if file_extension == '.pdf':
            
#             pdf_reader = PdfReader(file)
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
#         elif file_extension == '.txt':
#             text += file.read().decode("utf-8")
#         else:
#             st.warning(f"Unsupported file type: {file.name}. Files are not supported.")
#     return text

# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vectorstore(text_chunks):
#     embeddings = OpenAIEmbeddings()
#     embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore  



# def main():
#     load_dotenv()
#     st.set_page_config(page_title = "Document Centric Chatbot", page_icon =":books:")

#     st.header("Document Centric Chatbot :books:")
#     st.text_input("Ask a question about your documents:")
    
#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader("Upload your documents here and click on 'Process'", accept_multiple_files = True,type=['pdf', 'txt'])
#         if st.button("Process"):
#             with st.spinner("Processing"):
#                 # get document text
#                 raw_text = get_pdf_text(pdf_docs)
#                 st.write(raw_text)
#                 # get the text chunks
#                 text_chunks = get_text_chunks(raw_text)
                
#                 # create vector store
#                 vectorstore  = get_vectorstore(text_chunks)



# if __name__ == '__main__':
#     main()



# import streamlit as st
# from PyPDF2 import PdfReader
# import os
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings  # Local embeddings
# from langchain.vectorstores import FAISS

# def get_pdf_text(get_docs):
#     text = ""
#     for file in get_docs:
#         file_extension = os.path.splitext(file.name)[1].lower()
        
#         if file_extension == '.pdf':
#             pdf_reader = PdfReader(file)
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
#         elif file_extension == '.txt':
#             text += file.read().decode("utf-8")
#         else:
#             st.warning(f"Unsupported file type: {file.name}. Files are not supported.")
#     return text

# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vectorstore(text_chunks):
#     # Using local Ollama embeddings with Mistral (no API needed)
#     embeddings = OllamaEmbeddings(model="mistral")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore

# def main():
#     st.set_page_config(page_title="Document Centric Chatbot", page_icon=":books:")

#     st.header("Document Centric Chatbot :books:")
#     st.text_input("Ask a question about your documents:")
    
#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your documents here and click on 'Process'", 
#             accept_multiple_files=True,
#             type=['pdf', 'txt']
#         )
#         if st.button("Process"):
#             with st.spinner("Processing..."):
#                 # Get document text
#                 raw_text = get_pdf_text(pdf_docs)
#                 # Get the text chunks
#                 text_chunks = get_text_chunks(raw_text)
#                 # Create vector store
#                 vectorstore = get_vectorstore(text_chunks)
#                 st.success("Documents processed successfully!")

# if __name__ == '__main__':
#     main()

# import streamlit as st
# from PyPDF2 import PdfReader
# import os
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS

# def get_pdf_text(get_docs):
#     text = ""
#     for file in get_docs:
#         file_extension = os.path.splitext(file.name)[1].lower()
        
#         if file_extension == '.pdf':
#             pdf_reader = PdfReader(file)
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
#         elif file_extension == '.txt':
#             text += file.read().decode("utf-8")
#         else:
#             st.warning(f"Unsupported file type: {file.name}. Files are not supported.")
#     return text

# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vectorstore(text_chunks):
#     # Using local HuggingFace embeddings (no API needed)
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={'device': 'cpu'}  # Use 'cuda' if you have GPU
#     )
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore

# def main():
#     st.set_page_config(page_title="Document Centric Chatbot", page_icon=":books:")

#     st.header("Document Centric Chatbot :books:")
#     st.text_input("Ask a question about your documents:")
    
#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your documents here and click on 'Process'", 
#             accept_multiple_files=True,
#             type=['pdf', 'txt']
#         )
#         if st.button("Process"):
#             with st.spinner("Processing"):
#                 # get document text
#                 raw_text = get_pdf_text(pdf_docs)
#                 # get the text chunks
#                 text_chunks = get_text_chunks(raw_text)
#                 # create vector store
#                 vectorstore = get_vectorstore(text_chunks)
#                 st.success("Documents processed successfully!")

# if __name__ == '__main__':
#     main()

import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings  # Local embeddings
from langchain.vectorstores import FAISS

def get_pdf_text(get_docs):
    text = ""
    for file in get_docs:
        file_extension = os.path.splitext(file.name)[1].lower()
        
        if file_extension == '.pdf':
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file_extension == '.txt':
            text += file.read().decode("utf-8")
        else:
            st.warning(f"Unsupported file type: {file.name}. Files are not supported.")
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # Using local Ollama embeddings with Mistral (no API needed)
    embeddings = OllamaEmbeddings(model="mistral")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def main():
    st.set_page_config(page_title="Document Centric Chatbot", page_icon=":books:")

    st.header("Document Centric Chatbot :books:")
    st.text_input("Ask a question about your documents:")
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your documents here and click on 'Process'", 
            accept_multiple_files=True,
            type=['pdf', 'txt']
        )
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Get document text
                raw_text = get_pdf_text(pdf_docs)
                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # Create vector store
                vectorstore = get_vectorstore(text_chunks)
                st.success("Documents processed successfully!")

if __name__ == '__main__':
    main()