import requests
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
import os
import json

FMP_BASE_URL = "https://financialmodelingprep.com/stable"

load_dotenv()

FMP_API_KEY = os.environ.get("FMP_API_KEY")
financial_statements_db_path = Path("data/statements.json")


STATEMENT_TYPES = ["income-statement", "balance-sheet-statement", "cash-flow-statement"]

def collect_financial_statements(symbol: str, statement_required: str, limit:int=5):
    url = f"{FMP_BASE_URL}/{statement_required}?symbol={symbol}&limit={limit}&period=FY&apikey={FMP_API_KEY}" 
   
    try: 
        response = requests.get(url)

        financial_statements_data = response.json()

        documents_formated = []
        for fin_statement in financial_statements_data:
            retrieved_text = "\n".join([f"{k}:{v}" for k, v in fin_statement.items()])
            documents_formated.append({ "page_content":retrieved_text,
                "metadata":{
                    "date": fin_statement.get('date', "),
                    "corporation":symbol,
                    "statement": statement_required}})
    
        return documents_formated

    except Exception as e:
        print(f"Error retriving data for: {symbol}, error {e}")       


def retrieve_all_financial_statements(symbols:  List[str], limit: int=5):
    all_financial_statements = []
    for symbol in symbols:
        for financial_statement in STATEMENT_TYPES:
            documents = collect_financial_statements(symbol, financial_statement, limit=limit)
            all_financial_statements.extend(documents)
    return all_financial_statements
        
def save_statements_to_json(statements: List[Dict]):

    if financial_statements_db_path.exists():
        with open(financial_statements_db_path, 'r', encoding='utf-8') as f:
            financial_statements_db = json.load(f)
            financial_statements_db.extend(statements)
    
    else:
        financial_statements_db = []
        financial_statements_db.extend(statements)
    
    with open(financial_statements_db_path, 'w', encoding='utf-8') as f:
        json.dump(financial_statements_db, f, ensure_ascii=True, indent=2)
    
def get_financial_statements_db():
    with open(financial_statements_db_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    symbols = ['AAPL', 'MSFT', 'NVDA', 'AMZN']
    statements = retrieve_all_financial_statements(symbols, limit=2)
    print(f"Statements retrieved: {len(statements)}")

    save_statements_to_json(statements)

    financial_statements_db = get_financial_statements_db()