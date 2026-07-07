'''
interrupt() 적용 ? Human-in-the-loop (사람 승인 대기)
노드 실행 중간에 interrupt()를 호출하면 그래프 실행이 멈추고, 사람이 값을 입력해줄 때까지 대기합니다.
승인 절차나 위험한 작업 직전 확인에 사용합니다.
interrupt(value)를 호출하면 해당 지점에서 그래프가 멈추고 value가 사람에게 보여줄 정보로 반환됩니다.
중단된 그래프는 app.invoke(Command(resume=응답값), config=...) 형태로 재개합니다.
interrupt()는 체크포인터 없이는 동작하지 않습니다 ? 중단 시점의 상태를 저장해둬야 나중에 재개할 수 있기 때문입니다.
'''
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


class State(TypedDict):
    action: str
    approved: bool
    result: str


def propose_action(state: State) -> dict:
    print(f"[propose_action] 제안된 작업: {state['action']}")
    return {}


def request_approval(state: State) -> dict:
    # 여기서 그래프 실행이 멈추고, 사람의 입력을 기다림
    # interrupt()에 넘긴 값은 사람에게 보여줄 "질문 내용"으로 사용됨
    decision = interrupt(
        {"question": f"'{state['action']}' 작업을 실행해도 될까요? (yes/no)"}
    )
    print(f"[request_approval] 사람의 응답: {decision}")
    return {"approved": decision == "yes"}


def execute_or_cancel(state: State) -> dict:
    if state["approved"]:
        return {"result": f"'{state['action']}' 실행 완료"}
    return {"result": "작업이 취소되었습니다"}


graph = StateGraph(State)
graph.add_node("propose_action", propose_action)
graph.add_node("request_approval", request_approval)
graph.add_node("execute_or_cancel", execute_or_cancel)

graph.add_edge(START, "propose_action")
graph.add_edge("propose_action", "request_approval")
graph.add_edge("request_approval", "execute_or_cancel")
graph.add_edge("execute_or_cancel", END)

# interrupt()를 쓰려면 checkpointer가 반드시 필요 (중단된 지점부터 재개하기 위해)
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "approval-1"}}

# 1단계: 실행 -> request_approval 노드에서 interrupt 발생, 여기서 멈춤
result = app.invoke({"action": "프로덕션 DB 삭제", "approved": False, "result": ""}, config=config)
print("중단됨:", result["__interrupt__"])
# 중단됨: [Interrupt(value={'question': "'프로덕션 DB 삭제' 작업을 실행해도 될까요? (yes/no)"}, ...)]

# 2단계: 사람이 실제로 응답을 입력 (예: 콘솔 입력, 슬랙 승인 버튼 등)
final_result = app.invoke(Command(resume="yes"), config=config)
print("최종 결과:", final_result["result"])
# 최종 결과: 작업이 취소되었습니다