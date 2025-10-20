from langchain_openai import ChatOpenAI
from src.constants import default_llm_model       
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate

load_dotenv()

def get_llm(model: str, temperature: float):
    try:
        print(f"Getting {model} with {temperature} temp")
        return ChatOpenAI(model=model, temperature=temperature)
    except Exception as e:
        print(f'Error connecting to model: {e}')

def query_solution(query:str, temperature: float, model: str = default_llm_model):
    try:
        llm = get_llm(model, temperature)
        response = llm.invoke(query)
        return response.content

    except Exception as e:
        print(f"Error querying llm: {e}")

def query_rewritting(query:str, quantity:int = 3, model: str = default_llm_model):

    prompt_template = f"""
    You are a query rewriting assistant. Given a user's question, rewrite it into {quantity} semantically similar variations that could retrieve different relevant information from a vector database. 
    The rewrites should focus on the same meaning but use different wording or phrasing.

    Return questions separated exclusively by 1 comma, nothing else.

User query: "{query}"
    """

    try:
        llm = get_llm(model, 0.1)
        response = llm.invoke(prompt_template)
        return response.content.split(',')

    except Exception as e:
        print(f"Error querying llm: {e}")


