from datasets import Dataset
import json
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git
from ragas import evaluate
from src.constants import test_data_path, test_result_data_path, raga_result_data_path
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from dotenv import load_dotenv

load_dotenv()


def load_test_data(path:str= test_data_path):
    with open(path, "r", encoding='utf-8') as f:
        testing_data = json.load(f)
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


if __name__ == "__main__":
    
    dataset = load_test_results_data()

    results = run_ragas_evaluation(dataset)
    save_results(results, "ragas_report.json")
    print("Evaluation completed:", results)
