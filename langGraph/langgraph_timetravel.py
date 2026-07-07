'''
Time Travel : State에 저장된 특정 시점의 상태를 불러와서 그 시점 부터 다시 실행
'''
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    count: int
    log : list[str]

#subgraph 정의
def increment(state: State) -> dict:
    new_count = state["count"] + 1
    print("[increment] Step1 : count{state['count]} -> {new_count}")
    return {
        "count" : new_count,
        "log" : state["log"] + [f"count = {new_count}"]
    }

graph= StateGraph(State)
graph.add_node("increment", increment)

graph.add_edge(START, "increment")
graph.add_edge("increment", END)

checkpointer = MemorySaver()
config = {"configurable": {"thread_id": "tnut"}}
app = graph.compile(checkpointer=checkpointer)

app.invoke(
    {"count": 0, "log": []},
    config=config
)

for _ in range(4) :
    result = app.invoke({}, config = config)

print("현재상태 : ", app.get_state(config).values)

history = list(app.get_state_history(config))
for snapshot in history :
    print(f"count = {snapshot.values['count']} | checkpoint_id = {snapshot.config['configurable']['checkpoint_id']}")
    
target_snapshot = next(s for s in history if s.values['count'] == 2)
target_config = target_snapshot.config
print(f"되돌아갈 지점 : count = {target_snapshot.values['count']}")

result = app.invoke({}, config = target_config)
print("과거 시점에서 재실행한 결과 : ", result)

print()
print("재실행 후 전체 이력")
for snapshot in history :
    print(f"count = {snapshot.values['count']} | checkpoint_id = {snapshot.config['configurable']['checkpoint_id']}")