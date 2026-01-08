from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph
from pydantic import BaseModel

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm_model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)

# Definir estado do graph
class GRaphState(BaseModel):
    input: str
    output: str
    tipo: str = None

# Funcao de Realizar calculo
def realizar_calculo(state: GRaphState) -> GRaphState:
    return GRaphState(input=state.input,
                      output="Resposta do calculo : 42")

# Funçao para responder perguntas normais
def responder_curiosidades(state: GRaphState) -> GRaphState:
    response = llm_model.invoke([HumanMessage(content=state.input)])
    return GRaphState(input=state.input,
                      output=response.content)

# Funçao para tratar perguntas nao reconhecidas
def respoder_erro(state: GRaphState) -> GRaphState:
    return GRaphState(input=state.input,
                      output="Desculpa nao entendi sua pergunta")

# Funçao de classificaçao dos nodes
def classificar(state: GRaphState) -> GRaphState:
    pergunta = state.input.lower()
    if any(palavra in pergunta for palavra in ["soma", "quanto é", "+", "calcular"]):
        tipo = "calculo"
    elif any(palavra in pergunta for palavra in ["quem", "onde", "por que", "qual"]):
        tipo = "curiosidade"
    else:
        tipo = "desconhecido"

    return GRaphState(input=state.input,
                      output="",
                      tipo=tipo)


# Criando o graph e adicionando os nodes
graph = StateGraph(GRaphState)

graph.add_node("classificar", classificar)
graph.add_node("realizar_calculo", realizar_calculo)
graph.add_node("responder_curiosidades", responder_curiosidades)
graph.add_node("respoder_erro", respoder_erro)

# Adicionando condicionais
graph.add_conditional_edges(
    "classificar",
    lambda state: {
        "calculo": "realizar_calculo",
        "curiosidade": "responder_curiosidades",
        "desconhecido": "respoder_erro"
    }[state.tipo]
)

# Definindo enntrada e saida e compilaçao
graph.set_conditional_entry_point("classificar")
graph.set_finish_point("realizar_calculo", "responder_curiosidades", "respoder_erro")
export_graph = graph.compile()

# Testando o Projeto
if __name__ == "__name__":
    exemplos= [
        "Quanto é 10 + 5?",
        "Quem inventou a lampada?",
        "Me diga um comando espeecial"
    ]
    for exemplo in exemplos:
        result = export_graph.invoke(GRaphState(input=exemplo, output=""))
        print(f"Pergunta: {exemplo}\nResposta: {result['output']}\n")