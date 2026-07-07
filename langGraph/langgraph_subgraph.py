from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    text: str
    step_log : list[str]

#subgraph 정의
def sub_step1(state: State) -> dict:
    print("[Sub Graph] Step1 : 소문자 -> 대문자")
    return {
        "text" : state["text"].upper(),
        "step_log" : state["step_log"] + ["sub_step1"]
    }


def sub_step2(state: State) -> dict:
    print("[Sub Graph] Step2 : 접두어 추가")
    return {
        "text" : f"[처리됨].{state['text']}",
        "step_log" : state["step_log"] + ["sub_step2"]
    }

subpraph_builder = StateGraph(State)
subpraph_builder.add_node("sub_step1", sub_step1)
subpraph_builder.add_node("sub_step2", sub_step2)

subpraph_builder.add_edge(START, "sub_step1")
subpraph_builder.add_edge("sub_step1", "sub_step2")
subpraph_builder.add_edge("sub_step2", END)

subgraph = subpraph_builder.compile()

#parent graph 정의
def parent_start(state: State) -> dict:
    print("[Parent Graph] Start Node Execute")
    return {
        "step_log" : state["step_log"] + ["parent_start"]
    }

def parent_end(state: State) -> dict:
    print("[Parent Graph] End Node Execute")
    return {
        "step_log" : state["step_log"] + ["parent_end"]
    }

pgraph_builder= StateGraph(State)
pgraph_builder.add_node("parent_start", parent_start)
pgraph_builder.add_node("sub_workflow", subgraph)
pgraph_builder.add_node("parent_end", parent_end)

pgraph_builder.add_edge(START, "parent_start")
pgraph_builder.add_edge("parent_start", "sub_workflow")
pgraph_builder.add_edge("sub_workflow", "parent_end")
pgraph_builder.add_edge("parent_end", END)

parent_graph = pgraph_builder.compile()

result = parent_graph.invoke({"text": "", "step_log": []},)
print(result)