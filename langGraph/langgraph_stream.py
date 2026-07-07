from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

from operator import add
from dotenv import load_dotenv

load_dotenv()

#Annotated는 여러 노드의 로그를 누적하는데 사용하는 리듀서
class State(TypedDict):
    question: str
    answer : str
    log : Annotated[list[str], add]

def make_llm() -> ChatOpenAI :
    return ChatOpenAI(model = "gpt-4o-mini", temperature = 0, streaming = True)

def analyse(state: State) -> dict :
    "질문을 분석하는 Node"
    return {
        "log": [
            f"[analyze] 질문 분석 완료 -> {state['question']}"
        ]
    }

def generate(state: State) -> dict :
    "LLM을 호출하고 답변을 생성하는 Node"
    response = make_llm().invoke(HumanMessage(content= state['question']))
    return {"answer" : response.content, "log" : ["[generate]답변 생성 완료"]}

builder = StateGraph(State)
builder.add_node("analyse", analyse)
builder.add_node("generate", generate)

builder.add_edge(START, "analyse")
builder.add_edge("analyse", "generate")
builder.add_edge("generate", END)

app = builder.compile()

inputs = {"question" : "langgraph의 stream에 대해서 설명해 주세요.", "answer" : "", "log" : []}

def run_values() :
    print("Stream_mode = 'values' -> 스텝마다 '전체 State의 스냅샷' " )
    for step in app.stream(inputs, stream_mode = "values") :
        print(step)
    print

def run_updates() :
    print("Stream_mode = 'updates' -> 각 노드가 반환한 state의 변경된 데이터만 " )
    for step in app.stream(inputs, stream_mode = "updates") :
        print(step)
    print

def run_messages() :
    print("Stream_mode = 'messages' -> LLM '토큰단위'로 실시간 스트리밍 " )
    for chunk, metadata in app.stream(inputs, stream_mode = "messages") :
        node_name = metadata.get("langgraph_node")
        print(f"[node = {node_name} token = {chunk.content!r}]")
    print

if __name__ == "__main__" : 
    run_values()
    #run_updates()
    #run_messages()