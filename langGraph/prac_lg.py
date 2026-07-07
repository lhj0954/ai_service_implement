from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    message: str

def hello(state: State):
    return {
        "message": "Hello LangGraph"
    }

builder = StateGraph(State)

builder.add_node("hello", hello)
builder.add_edge(START, "hello")
builder.add_edge("hello", END)

graph = builder.compile()
result = graph.invoke( {"message":"" } )
print(result)