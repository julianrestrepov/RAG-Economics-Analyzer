from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from src.constants import default_llm_model
import numpy as np
from typing import List


# General pdf splitter
def pdf_splitter(loaded_pdfs: Document, chunk_size: int =800, chunk_overlap:int =100, model_name:str =default_llm_model) -> list[Document]:
    encoding = tiktoken.encoding_for_model(model_name)
    average_tokens = []

    # lenght function by tokens
    def number_tokens(text): 
        average_tokens.append(len(encoding.encode(text)))
        return len(encoding.encode(text))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   length_function = number_tokens)

    pdfs_chunks_list = []    
    for pdfs in loaded_pdfs:

        pdfs_chunks_list.extend(text_splitter.split_documents(pdfs))

    return pdfs_chunks_list


# split financial statements from FMP
def financial_statements_splitter(financial_statements: List[dict], chunk_size: int =800, chunk_overlap:int =100, model_name:str =default_llm_model):

    financial_statements_documents = [Document(page_content=statement['page_content'], metadata=statement['metadata']) for statement in financial_statements]

    encoding = tiktoken.encoding_for_model(model_name)

    def number_tokens(text): 
        return len(encoding.encode(text))
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   length_function = number_tokens,
                                                   separators=["\n\n", "\n", " "])
    
    chunked_statements = text_splitter(financial_statements_documents)

    return chunked_statements