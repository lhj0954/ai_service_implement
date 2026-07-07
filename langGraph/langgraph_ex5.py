'''
Checkpointer 적용 ? 상태 저장/재개 (memory)
체크포인터를 붙이면 그래프 실행 중 상태가 자동으로 저장되어, 나중에 같은 thread_id로 이어서 실행하거나 중간 상태를 조회할 수 있습니다
'''

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    messages: list[str]


def add_message(state: State) -> dict:
    new_msg = f"메시지 #{len(state['messages']) + 1}"
    print(f"[add_message] {new_msg} 추가")
    return {"messages": state["messages"] + [new_msg]}


graph = StateGraph(State)
graph.add_node("add_message", add_message)
graph.add_edge(START, "add_message")
graph.add_edge("add_message", END)

# 체크포인터 지정: 실행 중 상태를 메모리에 저장 (실무에서는 SqliteSaver/PostgresSaver 사용)
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# thread_id로 대화/세션을 구분 ? 같은 thread_id면 이전 상태 위에 이어서 실행됨
config = {"configurable": {"thread_id": "user-123"}}

app.invoke({"messages": []}, config=config)
app.invoke({}, config=config)  # 이전 상태를 이어받아 누적
result = app.invoke({}, config=config)

print(result["messages"])
# ['메시지 #1', '메시지 #2', '메시지 #3']

# 저장된 체크포인트 이력 조회 (타임 트래블 디버깅의 기반)
for snapshot in app.get_state_history(config):
    print(snapshot.values, "| next:", snapshot.next)