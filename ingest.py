import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

class ingestor:

    def __init__(self) -> None:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    #the data from pdfs is read and stored as texts
    
    def extract_pdf_text(self,pdfs):
        text = ""
        for pdf in pdfs:
            reader=PdfReader(pdf)
            for page in reader.pages:
                text = text + page.extract_text()
        return text
    
    def extract_url_text(self,urls):
        text = ""
        loaders = UnstructuredURLLoader(urls=urls)
        docs = loaders.load()
        for doc  in docs:
            text = text + doc.page_content
        return text

    
    '''
    
    #the data from urls are read and stored as documents
    
    def extract_url_text_chunks(self,urls):
        text = ""
        loaders = UnstructuredURLLoader(urls=urls)
        text = loaders.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap=100)
        chunks = text_splitter.split_documents(text)
        return chunks

    def store_vectors_urls(self,text_chunks,faiss_path="FAISS_store"):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        faiss_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        faiss_store.save_local(faiss_path)

    '''
    def extract_text_chunks(self,text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        return chunks


    def store_vectors(self,text_chunks,faiss_path="FAISS_store"):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        faiss_store = FAISS.from_texts(text_chunks,embedding=embeddings)
        faiss_store.save_local(faiss_path)
