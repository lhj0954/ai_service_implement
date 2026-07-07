#사용자의 요청 유형에 따라 다른 처리 노드로 보내기 예제
#라우터 함수는 State를 받아 다음 노드 이름(키)을 반환하고, LangGraph는 매핑 딕셔너리를 참조해 실제 노드로 이동합니다.
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    query: str
    category: str
    answer: str


def classify(state: State) -> dict:
    q = state["query"].lower()
    if "환불" in q or "취소" in q:
        category = "refund"
    elif "배송" in q:
        category = "shipping"
    else:
        category = "general"
    print(f"[classify] 분류 결과: {category}")
    return {"category": category}


def handle_refund(state: State) -> dict:
    return {"answer": "환불 절차 안내를 시작합니다."}


def handle_shipping(state: State) -> dict:
    return {"answer": "배송 조회 페이지로 안내합니다."}


def handle_general(state: State) -> dict:
    return {"answer": "일반 문의 담당자에게 연결합니다."}


# 라우터 함수: state를 보고 "다음에 실행할 노드 이름"을 문자열로 반환
#Literal은 변수나 파라미터가 특정 값들 중 하나만 가질 수 있다고 타입 힌트로 설정(선언)
def route_by_category(state: State) -> Literal["refund", "shipping", "general"]:
    return state["category"]


graph = StateGraph(State)
graph.add_node("classify", classify)
graph.add_node("refund", handle_refund)
graph.add_node("shipping", handle_shipping)
graph.add_node("general", handle_general)

graph.add_edge(START, "classify")

# classify 노드 다음, route_by_category()의 반환값에 따라 분기
# {라우터 반환값: 실제 노드 이름} 매핑을 명시적으로 지정
graph.add_conditional_edges(
    "classify",
    route_by_category,
    {
        "refund": "refund",
        "shipping": "shipping",
        "general": "general",
    },
)

graph.add_edge("refund", END)
graph.add_edge("shipping", END)
graph.add_edge("general", END)

app = graph.compile()

#print(app.invoke({"query": "배송이 언제 오나요?"}))
print(app.invoke({"query": "이거 환불하고 싶어요"}))