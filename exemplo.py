from humanfriendly.terminal import output
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm_model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)

# Definir do StateGraph
class GraphState(BaseModel):
    input: str
    output: str

# Funçao de resposta
def responder(state):
    input_message = state.input
    response = llm_model.invoke([HumanMessage(content=input_message)])
    return GraphState(input=state.input, output=response.content)

# Criar o Graph
graph = StateGraph(GraphState)
graph.add_node("responder", responder)
graph.set_entry_point("responder")
graph.set_finish_point("responder")

# Compilando o Graph
export_graph = graph.compile()

# Testar o agent
if __name__ == "__main__":
    result = export_graph.invoke(GraphState(input="Quem descobriu a america", output=""))
    print(result)
    
    # Visualizar o Grafo
    print(export_graph.get_graph().draw_mermaid())