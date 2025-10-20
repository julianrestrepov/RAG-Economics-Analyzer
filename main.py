from src.data_preprocessing import pdf_splitter
from src.loaders import load_pdfs
from src.embedders import chroma_embedder, get_chroma_embeddings
from src.llm_model import query_solution
from src.constants import PDF_DATA_FOLDER_DIR, chroma_database, test_data_path, DATA_FOLDER_DIR
from src.evaluator_ragas import load_test_data
import os

import json
from datasets import Dataset


if os.path.exists(chroma_database):
    vector_db = get_chroma_embeddings()
else:
    pdf_docs = load_pdfs(path=PDF_DATA_FOLDER_DIR)

    chunks_pdf = pdf_splitter(pdf_docs, chunk_size=200, chunk_overlap=0)

    vector_db = chroma_embedder(chunks_pdf)


def retrieve_context(question:str, k=5, return_list=False):
    retriever = vector_db.as_retriever(search_kwargs={"k":k})
    chunks_retrieved = retriever.get_relevant_documents(question)
    if not return_list:
        context = "\n\n".join([chunk.page_content for chunk in chunks_retrieved])
    else:
        context =  [chunk.page_content for chunk in chunks_retrieved]
    return context



result_data = []
if __name__ == "__main__":
    question = input("Insert your question: ")
    context = retrieve_context(question, return_list=True)
    answer = query_solution(question, context)
    print(answer)
    for context_text in context:
        print(context.index(context_text))
        print(context_text)
    print()
    print()
    

    """  testing_data = load_test_data()

        for question_line in testing_data:
            context = retrieve_context(question_line['question'])
            answer = query_solution(question_line['question'], context)

            result_data.append({
                "question":question_line['question'],
                "retrieved_contexts": [context],
                "answer":answer,
                "ground_truth": question_line['ground_truth']
            })

            print(f"Question: {question_line['question']}")
            #print(f"Context: {context}")
            print(f"Answer: {answer}")
            print(f"Truth: {question_line['ground_truth']}")
            print()
        
        with open(os.path.join(DATA_FOLDER_DIR, 'test_results.json'), 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
    """