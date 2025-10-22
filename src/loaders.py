from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from typing import List
from src.constants import PDF_DATA_FOLDER_DIR
import os
import re


def general_pdf_text_cleaning(document_to_clean: Document):

    # text cleaning
    for page in document_to_clean:
        page.page_content = re.sub(r"<[^>]+>|http\S+|www\.\S+|\S+@\S+", " ", page.page_content) # removes html, url and emails
        page.page_content = re.sub(r"[\x00-\x1f]+", " ", page.page_content) # symbols
        page.page_content = re.sub(r"\s{2,}", " ", page.page_content)
    return document_to_clean

# find all pdfs from data/pdfs folder
def find_pdfs():
    return [os.path.join(PDF_DATA_FOLDER_DIR, f) for f in os.listdir(PDF_DATA_FOLDER_DIR) if f.lower().endswith(".pdf")]


def load_pdfs() -> List[Document]:

    pdfs_path = find_pdfs()

    cleaned_sliced_pdfs = []
    for pdf_file in pdfs_path:
        loader = PyPDFLoader(pdf_file)
        raw_pdf = loader.load()

        cleaned_sliced_pdfs.append(general_pdf_text_cleaning(raw_pdf))
    return cleaned_sliced_pdfs
