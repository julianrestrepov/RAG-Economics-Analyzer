from langchain_openai import ChatOpenAI
from src.constants import default_llm_model       
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
import os
import requests
import json

load_dotenv()


"Returns LLM model"
def get_llm(model: str, temperature: float):
    try:
        return ChatOpenAI(model=model, temperature=temperature)
    except Exception as e:
        print(f'Error connecting to model: {e}')


"LLM to answer user question based on provided information"
def query_solution(query:str, context:str, conversation_history:str, temperature: float, model: str = default_llm_model, fred_data: str=None):
    query_template = f"""
        You are an economic analyst.
        Use only the available information in <context>, <previous conversations> and <FRED Data> to answer the <User Question>.
        You may synthesize and reason within the available information to explain relationships, comparison or trends. 
        if available information can't answer the question respond "Sorry, my context window doesn't contain this information"

        Rules:
        - Never use outside knowledge
        - Answer concisely and factually.
        - Use neutral financial language.

        Previous Conversations:
        {conversation_history}

        User Question:
        {query}

        Context:
        {context}

        FRED Data:
        {fred_data}
        """
    
    query_input = query_template.format(
            conversation_history=conversation_history,
            query=query,
            context=context,
            fred_data= fred_data
        )

    try:
        llm = get_llm(model, temperature)
        response = llm.invoke(query_input)
        print("PROMPT: ", query_input, "\n\n")
        return response.content

    except Exception as e:
        print(f"Error querying llm: {e}")


"LLM to rewrite 1 query into x amount of semantically similar versions"
def query_rewritting(query:str, quantity:int = 3, model: str = default_llm_model):

    prompt_template = f"""
    You are a semantic query rewriting system. Given a user question, rewrite it into {quantity} semantically similar variations that could retrieve different relevant information from a vector database. 
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


"LLM to determinate if FRED API is required and builts request to API."
def query_fred_api_needed(query:str, model: str = default_llm_model):
    prompt = f"""
        You are a routing assistant for an economics RAG system.
        Decide if the user's question requires calling the FRED API.
        If yes, return a JSON with:
        - "needs_fred": true
        - "fred_series_id": the best FRED series (use exact code if known)
        - "search_text": if series_id unknown
        - "start_date": (YYYY-MM-DD)
        - "end_date": (YYYY-MM-DD)
        if not then return "needs_fred":false

        User query: {query}
        """
    
    try:
        llm = get_llm(model, 0.1)
        response = llm.invoke(prompt)
        response_json = json.loads(response.content)

        if response_json['needs_fred']:
            request_template = {
                'series_id':response_json.get('fred_series_id'),
                "file_type": "json",
                'api_key': os.getenv("FRED_API_KEY"),
                'observation_start':response_json.get('start_date'),
                'observation_end':response_json.get('end_date'),

            }
            r = requests.get("https://api.stlouisfed.org/fred/series/observations", params=request_template)
            fred_data_dict = r.json()
            fred_data_string = json.dumps(fred_data_dict["observations"], indent=2)
            fred_data_string = fred_data_string.replace("{", "{{").replace("}", "}}")
            return fred_data_string
        
        else:
            return False

    except Exception as e:
        print(f"Error querying llm: {e}")