from langgraph.graph import StateGraph, START, END
from typing import TypedDict

#데이터 저장소 State 객체
class State(TypedDict) : 
    text : str

#작업 단위 node
def step1_upppercase(state: State) -> dict :
    return {"text" : state["text"].upper()}

def step2_add_exclaim(state: State) -> dict :
    return {"text" : state["text"] + "!!!"}

def step3_wrap(state: State) -> dict :
    return {"text" : f"({state['text']})"}

#그래프 설계/생성
graph = StateGraph(State)

graph.add_node("upper", step1_upppercase)
graph.add_node("add_claim", step2_add_exclaim)
graph.add_node("wrap", step3_wrap)

graph.add_edge(START, "upper")
graph.add_edge("upper", "add_claim")
graph.add_edge("add_claim", "wrap")
graph.add_edge("wrap", END)

app = graph.compile()
result = app.invoke({"text":"hello langgraph"})
print(result)
