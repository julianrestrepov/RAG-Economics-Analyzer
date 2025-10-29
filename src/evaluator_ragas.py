from datasets import Dataset
import json
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git
from ragas import evaluate
from src.constants import test_data_path, test_result_data_path, raga_result_data_path
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from dotenv import load_dotenv
import pandas as pd
from src.query import VectorRetriever
from src.llm_model import query_solution
load_dotenv()


def evaluate_model(retriever: VectorRetriever, top_k, model, temperature:float):
    print(f"Running RAGAS Evaluation with:  \n temp:{temperature} \n model:{model} \n top_k:{top_k}")

    test_data = load_test_data()
    results_input = []
    for i, test_item in enumerate(test_data):
        print(f"Testing #{i}: ")
        query = test_item['question']
        print(f"query: {query}")
        ground_truth = test_item['ground_truth']
        print(f"ground_truth: {ground_truth}")
        context_retrieved = retriever.single_query(query, top_k, return_list=True)
        answer = query_solution(query, context_retrieved, temperature=temperature, model=model)
        print(f"answer: True")
        results_input.append({
            "question": query,
            "answer": answer,
            "contexts": context_retrieved,
            "ground_truth": ground_truth
        })
        print("\n")

    results_dataset = Dataset.from_list(results_input)
    results_ragas = run_ragas_evaluation(results_dataset)

    save_results(results_ragas, "ragas_report.json")

    df = results_ragas.to_pandas()

    scores = {
    "faithfulness": df["faithfulness"].mean(),
    "answer_relevancy": df["answer_relevancy"].mean(),
    "context_precision": df["context_precision"].mean(),
    "context_recall": df["context_recall"].mean()
}

    return scores


def load_test_data(path:str= test_data_path, questions_to_test:int = 5):
    with open(path, "r", encoding='utf-8') as f:
        testing_data = json.load(f)
        testing_data = testing_data[:questions_to_test] # limiting data to run for API Cost
    return Dataset.from_list(testing_data)

def load_test_results_data(path:str= test_result_data_path):
    with open(path, "r", encoding='utf-8') as f:
        testing_data = json.load(f)
    for item in testing_data:
        if not isinstance(item['retrieved_contexts'], list):
            contexts = [item['retrieved_contexts']]
            item["retrieved_contexts"] = contexts
    return Dataset.from_list(testing_data[60:])


def save_results(results, path: str=raga_result_data_path):
    results_df = results.to_pandas()
    results_df.to_csv(path)

def run_ragas_evaluation(dataset: Dataset):
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    return evaluate(dataset=dataset, metrics=metrics)

