from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_core.tools import tool
import os

from exemplo import export_graph

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm_model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)

# Defini o prompt do sistema
system_message = SystemMessage(
    """
    Voce é um assistente. Se o usuario pedir contas, use a ferramenta "somar". Caso contrario
    apenas responda normalmente
    """
)

# Definindo a ferramenta de soma
@tool("somar")
def somar(valores: str) -> str:
    """ Soma dois numeros separados por virgula"""
    try:
        a,b = map(float, valores.split(","))
        return str(a + b)
    except Exception as e:
        return f"Erro ao somar: {str(e)}"
    
# Criaçao do agente com LagGraph
tools = [somar]
graph = create_react_agent(
    model=llm_model,
    tools=tools,
    prompt=system_message
)
export_graph = graph

# Extrai a resposta final
def extrair_resposta_final(result):
    ai_massege = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
    if ai_massege:
        return ai_massege[-1].content
    else:
        return "Nenhuma mensagem final encontrada"

# Testar o agente
if __name__ == "__name__":
    entrada1 = HumanMessage(content="Quanto é 8 + 5?")
    result1 = export_graph.invoke({"messages": [entrada1]})
    for m in result1["messages"]:
        print(m)
    resposta_texto_1 = extrair_resposta_final(result1)
    print("Resposta 1: ", resposta_texto_1)

    print("------------------")

    entrada2 = HumanMessage(content="Quem pintou a monalisa")
    result2 = export_graph.invoke({"messages": [entrada2]})
    for m in result2["messages"]:
        print(m)
    resposta_texto_2 = extrair_resposta_final(result2)
    print("Resposta 2: ", resposta_texto_2)