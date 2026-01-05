from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from dotenv import load_dotenv

from exemplo2_tool import system_message

# Configs
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)

# Prompt do sistema
system_message = SystemMessage(
    content= """
    Vocec é um pesquisado muito sarcastico e ironico.
    Use ferramenta 'search' sem que nescessario, especialmente para perguntas
    que exigem informaçoes da web
    """
)

# Criando a ferramenta SEARCH
@tool("search")
def search_web(query: str = "") -> str:
    """
    Busca informações na web baseada na consulta fornecida.
    
    Args:
        query: Termos para buscar dados na web
        
    Returns: 
        As informações encontradas na web ou uma mensagem indicando
        que nenhuma informação foi encontrada.
    """
    tavily_search = TavilySearchResults(max_results=3)
    return tavily_search.invoke(query)
    
# 4 - Criação do agente ReAct
tools = [search_web]
graph = create_react_agent(
    model, 
    tools=tools,
    prompt=system_message
)

export_graph = graph