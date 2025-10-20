from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from typing import List
from src.constants import PDF_DATA_FOLDER_DIR
import os
import re


def general_pdf_text_cleaning(document_to_clean: Document):
    for page in document_to_clean:
        page.page_content = re.sub(r"<[^>]+>|http\S+|www\.\S+|\S+@\S+", " ", page.page_content) # removes html, url and emails
        page.page_content = re.sub(r"[\x00-\x1f]+", " ", page.page_content) # symbols
        page.page_content = re.sub(r"\s{2,}", " ", page.page_content)
    return document_to_clean


def find_pdfs(path:str):
    return [os.path.join(PDF_DATA_FOLDER_DIR, f) for f in os.listdir(PDF_DATA_FOLDER_DIR) if f.lower().endswith(".pdf")]


def load_pdfs(path: str) -> List[Document]:

    if path == PDF_DATA_FOLDER_DIR:
        path = find_pdfs(path)

    cleaned_sliced_pdfs = []
    for pdf_file in path:
        print(f"Loading file: {pdf_file.split('\\')[-1]}")
        loader = PyPDFLoader(pdf_file)
        raw_pdf = loader.load()

        cleaned_sliced_pdfs.append(general_pdf_text_cleaning(raw_pdf))
    print()
    return cleaned_sliced_pdfs
    