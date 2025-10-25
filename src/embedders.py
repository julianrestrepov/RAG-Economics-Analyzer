from langchain_chroma import Chroma
from typing import List
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings


class ChromaDatabase:

    "Initializes retriever with embedding model."
    def __init__(self, embedding_model:str, distance_metric:str):
        self.embedding_model = OpenAIEmbeddings(model=embedding_model)
        self.distance_metric = distance_metric

    "Returns vector db."
    def get_vector_db(self):
        return self.vector_db

    "Resets vector db."
    def erase_chroma_db(self):
        self.vector_db = None

    "Transforms text chunks into embeddings and stores in vector db."
    def embed_documents(self, chunks: List[Document], batch_size:int=100):

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]

            if self.vector_db is None:
                self.vector_db = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embedding_model
                )
            else:
                self.vector_db.add_documents(batch)