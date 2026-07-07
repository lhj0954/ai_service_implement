'''
LangGraph 직선형 Demo 코드
'''
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

#데이터 저장소 State 객체
class Mystate(TypedDict) : 
    message : str

#작업 단위 node
def say_hello(state) :
    return {"message" : "Hello, LangGraph~"}

#그래프 설계/생성
graph = StateGraph(Mystate)

graph.add_node("hello", say_hello)
graph.add_edge(START, "hello")
graph.add_edge("hello", END)

app = graph.compile()
result = app.invoke({"message":""})
print(result)
