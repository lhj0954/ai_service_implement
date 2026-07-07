'''
하나의 노드가 매번 "조건 충족 여부"를 검사해서, 충족되지 않으면 자기 자신으로 돌아가고, 충족되면 다음 노드로 넘어갑니다.
add_conditional_edges의 매핑에서 출발 노드와 도착 노드를 동일하게 지정하면 반드시 종료 조건(최대 시도 횟수 등)을 함께 넣어 무한 루프를 방지해야 합니다.
'''
import random
from typing import TypedDict

class State(TypedDict):
    attempt: int
    score: int
    result: str


def guess_and_score(state: State) -> dict:
    attempt = state["attempt"] + 1
    score = random.randint(1, 100)  # 실제로는 LLM 평가 점수 등이 들어갈 자리
    print(f"[시도 {attempt}] 점수: {score}")
    return {"attempt": attempt, "score": score}


def finalize(state: State) -> dict:
    return {"result": f"{state['attempt']}번째 시도에서 통과 (점수 {state['score']})"}


# 조건 충족 여부를 판단해 다음 노드를 결정하는 라우터
def check_condition(state: State) -> str:
    if state["score"] >= 80:
        print(" -> 조건 충족, 다음 단계로 진행")
        return "finalize"
    if state["attempt"] >= 5:
        print(" -> 최대 시도 횟수 도달, 강제 종료")
        return "finalize"
    print(" -> 조건 미충족, 다시 시도 (self-loop)")
    return "retry"


from langgraph.graph import StateGraph, START, END

graph = StateGraph(State)
graph.add_node("guess_and_score", guess_and_score)
graph.add_node("finalize", finalize)

graph.add_edge(START, "guess_and_score")

# 같은 노드("guess_and_score")로 되돌아갈 수도 있고, finalize로 빠져나갈 수도 있음
graph.add_conditional_edges(
    "guess_and_score",
    check_condition,
    {
        "retry": "guess_and_score",  # 자기 자신을 다시 가리킴 (self-loop)
        "finalize": "finalize",
    },
)

graph.add_edge("finalize", END)

app = graph.compile()
result = app.invoke({"attempt": 0, "score": 0, "result": ""})
print(result["result"])