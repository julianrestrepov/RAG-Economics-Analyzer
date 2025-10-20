import os

default_llm_model = 'gpt-4o-mini'
MODELS_AVAILABLE = ['gpt-4o-mini', 'gpt-4-turbo']
default_embedding_model = 'text-embedding-3-small'


pdfs_folder = r'C:\Users\julia\Desktop\RAG Project\data\pdfs'
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER_DIR = os.path.join(PROJECT_DIR, "data")
PDF_DATA_FOLDER_DIR = os.path.join(DATA_FOLDER_DIR, "pdfs")

test_data_path = os.path.join(DATA_FOLDER_DIR, "test_data.json")
test_result_data_path = os.path.join(DATA_FOLDER_DIR, "test_results.json")
raga_result_data_path = os.path.join(DATA_FOLDER_DIR, "raga_results.csv")