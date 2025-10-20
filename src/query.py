from langchain_community.vectorstores import Chroma
from collections import defaultdict
import re



class VectorRetriever:
    def __init__(self, vector_db):
        self.vector_db = vector_db


    def single_query(self, query, top_k):
        return self.vector_db.similarity_search(query=query, k=top_k)

    def multiple_query(self, queries:[str], top_k: int = 3):

        chunk_dict_db = defaultdict(lambda: {"count":0, "max_score":0})
        for query in queries:
                retrieved_chunks = self.vector_db.similarity_search_with_score(query=query, k=top_k)
                for chunk, score in retrieved_chunks:
                    key = chunk.metadata.get("source", chunk.page_content[:50])  # use ID or first chars
                    d = chunk_dict_db[key]
                    d["count"] += 1
                    d["max_score"] = max(d["max_score"], score)
                    d["text"] = chunk.page_content

        # Rank by count then score
        sorted_chunks = sorted(chunk_dict_db.values(), key=lambda x: (x["count"], x["max_score"]), reverse=True)

        # Clean and join
        def clean(t):
            return re.sub(r"[^\x00-\x7F]+", " ", t).strip()

        top_docs = [clean(x["text"]) for x in sorted_chunks[:top_k]]
        context = "\n\n".join(top_docs)

        return context, sorted_chunks
