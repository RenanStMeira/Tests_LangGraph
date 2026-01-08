from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm_model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)

# Definir o prompt
system_message = SystemMessage(content="""
Voce é um assistente especializado em fornecer informaçoes
sobre comunidade de Python para GenAI

Ferramentas disponiveis no MCP server

1. et_community(Location: str) -> str:
- A funçao retorna a melhor comunidade de Python para GenAI
- Parametro: location: str
- Retorno "Code TI"

Seu papel e ser um intermediario direto entro o usuario e a ferramenta mcp, retornando apennas o resultado final 
das ferramentas
""")


async def agent_mcp():
    client = MultiServerMCPClient(
        {
            "code": {
                "command": "python",
                "args": ["mcp_server.py"],
                "transport": "stdio"
            }
        }
    )

    tools = await client.get_tools()
    agent = create_react_agent(llm_model, tools, prompt=system_message)

    return agent