"""
두 가지 방식을 SSE(Server-Sent Events)로 비교합니다.
  1) /api/agent/stream      : langgraph.prebuilt.create_react_agent   (블랙박스 Agent)
  2) /api/langgraph/stream  : StateGraph 직접 구성                     (투명한 State/노드/엣지)
 
"""

import json
import os
import re
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

load_dotenv()

app = FastAPI()


# ════════════════════════════════════════════════
#  공통 Tool 정의 (두 방식에서 동일하게 사용)
# ════════════════════════════════════════════════
@tool
def calculator(expression: str) -> str:
    """수학 계산을 수행합니다. 사칙연산(+,-,*,/)을 처리합니다.

    Args:
        expression: 계산할 수식 (예: 15 * 23 + 7)
    """
    try:
        sanitized = re.sub(r"[^0-9+\-*/().\s]", "", expression)
        result = eval(sanitized, {"__builtins__": {}}, {})  # 사칙연산만 남긴 뒤 계산
        return f"계산 결과: {expression} = {result}"
    except Exception:
        return f"계산 오류: {expression}을 계산할 수 없습니다."


WEATHER_DATA = {
    "서울": "맑음, 22°C, 습도 55%",
    "부산": "흐림, 24°C, 습도 70%",
    "제주": "비, 20°C, 습도 85%",
    "도쿄": "맑음, 25°C, 습도 50%",
}


@tool
def get_weather(city: str) -> str:
    """도시의 날씨 정보를 조회합니다.

    Args:
        city: 날씨를 조회할 도시명
    """
    return f"[{city}] 날씨: {WEATHER_DATA.get(city, '정보 없음 (기본: 맑음 20°C)')}"


tools = [calculator, get_weather]
tool_map = {t.name: t for t in tools}

SYSTEM_PROMPT = "당신은 계산과 날씨 조회를 도와주는 AI입니다. 한국어로 답변하세요."


def sse(data: dict) -> str:
    """dict를 SSE(Server-Sent Events) 포맷 문자열로 변환"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ════════════════════════════════════════════════
#  방식 1: LangChain Agent (create_react_agent)
#  — 단일 블랙박스: 입력 → Agent → 출력
#  — 내부 루프가 자동 관리됨
#  — 상태 흐름이 숨겨져 있음
# ════════════════════════════════════════════════
@app.get("/api/agent/stream")
async def agent_stream(question: str = Query(...)):
    async def event_generator():
        try:
            yield sse({"type": "arch", "step": "입력", "desc": "HumanMessage → create_react_agent"})

            model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

            # create_react_agent: 내부적으로 ReAct 루프를 자동 구성
            # 개발자는 내부 상태(State) 구조를 직접 볼 수 없음
            agent = create_react_agent(model=model, tools=tools, prompt=SYSTEM_PROMPT)

            yield sse({"type": "arch", "step": "Agent 내부 루프", "desc": "Reason → Act → Observe (자동 반복)"})

            execution_trace = []
            final_answer = ""

            async for update in agent.astream(
                {"messages": [HumanMessage(content=question)]},
                stream_mode="updates",
            ):
                # agent 노드: LLM 사고 또는 최종 답변
                if "agent" in update:
                    for msg in update["agent"].get("messages", []):
                        if getattr(msg, "tool_calls", None):
                            for tc in msg.tool_calls:
                                trace_item = {"node": "agent → Reason", "tool": tc["name"], "args": tc["args"]}
                                execution_trace.append(trace_item)
                                yield sse({"type": "trace", **trace_item})
                        if isinstance(msg.content, str) and msg.content.strip():
                            final_answer = msg.content

                # tools 노드: 도구 실행 결과
                if "tools" in update:
                    for msg in update["tools"].get("messages", []):
                        content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content, ensure_ascii=False)
                        trace_item = {"node": "tools → Act+Observe", "tool": msg.name, "result": content}
                        execution_trace.append(trace_item)
                        yield sse({"type": "trace", **trace_item})

            yield sse({"type": "arch", "step": "출력", "desc": "최종 AIMessage 반환"})
            yield sse({"type": "answer", "content": final_answer, "trace": execution_trace})
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ════════════════════════════════════════════════
#  방식 2: LangGraph (StateGraph)
#  — 명시적 노드 + 엣지 정의
#  — 각 단계의 상태(State)가 투명하게 보임
#  — 분기·조건·체크포인팅 직접 제어 가능
# ════════════════════════════════════════════════
def add_reducer(a, b):
    return (a or []) + b


def sum_reducer(a, b):
    return (a or 0) + b


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    tools_used: Annotated[list, add_reducer]
    step_count: Annotated[int, sum_reducer]
    final_output: str


@app.get("/api/langgraph/stream")
async def langgraph_stream(question: str = Query(...)):
    async def event_generator():
        try:
            yield sse({
                "type": "arch", "step": "State 초기화",
                "desc": "GraphState = { messages, tools_used, step_count, final_output }",
            })

            model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            bound_model = model.bind_tools(tools)

            # 노드 내부에서 만든 이벤트를 잠시 담아두는 큐.
            # (Python langgraph의 astream은 "노드가 끝난 뒤" 결과를 넘겨주므로,
            #  노드 실행 도중의 세부 이벤트는 여기 모았다가 해당 스텝 결과와 함께 순서대로 내보냅니다.)
            pending_events: list[str] = []

            # ── 노드 1: llm_node ─────────────────────────
            async def llm_node(state: GraphState) -> dict:
                pending_events.append(sse({
                    "type": "node", "name": "llm_node",
                    "input_msgs": len(state["messages"]),
                    "desc": "LLM에게 메시지 전달 → AIMessage 생성",
                }))

                system_msg = SystemMessage(content=SYSTEM_PROMPT)
                response = await bound_model.ainvoke([system_msg, *state["messages"]])

                pending_events.append(sse({
                    "type": "state_update", "node": "llm_node",
                    "has_tool_calls": bool(getattr(response, "tool_calls", None)),
                    "tool_calls": [t["name"] for t in (response.tool_calls or [])],
                    "content_preview": (response.content or "")[:80],
                }))

                return {
                    "messages": [response],
                    "step_count": 1,
                    "final_output": response.content or state.get("final_output", ""),
                }

            # ── 노드 2: tool_node ─────────────────────────
            async def tool_node(state: GraphState) -> dict:
                last_msg = state["messages"][-1]
                tool_calls = last_msg.tool_calls or []
                tool_results = []

                pending_events.append(sse({
                    "type": "node", "name": "tool_node",
                    "desc": f"{len(tool_calls)}개 Tool 실행",
                }))

                for tc in tool_calls:
                    result = await tool_map[tc["name"]].ainvoke(tc["args"])
                    tool_results.append({
                        "type": "tool", "tool_call_id": tc["id"], "name": tc["name"], "content": result,
                    })
                    pending_events.append(sse({
                        "type": "state_update", "node": "tool_node",
                        "tool": tc["name"], "args": tc["args"], "result": result,
                    }))

                return {
                    "messages": tool_results,
                    "tools_used": [t["name"] for t in tool_results],
                    "step_count": 1,
                }

            # ── 조건부 엣지: 계속할지 끝낼지 결정 ─────
            def should_continue(state: GraphState) -> str:
                last_msg = state["messages"][-1]
                has_tool_calls = bool(getattr(last_msg, "tool_calls", None))
                decision = "tools" if has_tool_calls else END

                pending_events.append(sse({
                    "type": "edge", "from": "llm_node", "to": decision,
                    "reason": "tool_calls 있음 → tool_node로 분기" if has_tool_calls else "tool_calls 없음 → END (완료)",
                }))

                return decision

            # ── 그래프 구성 ──────────────────────────────
            graph = StateGraph(GraphState)
            graph.add_node("llm_node", llm_node)
            graph.add_node("tool_node", tool_node)
            graph.add_edge(START, "llm_node")
            graph.add_conditional_edges("llm_node", should_continue, {"tools": "tool_node", END: END})
            graph.add_edge("tool_node", "llm_node")  # tool 실행 후 다시 llm으로

            checkpointer = MemorySaver()
            compiled = graph.compile(checkpointer=checkpointer)

            yield sse({
                "type": "arch", "step": "그래프 구조",
                "desc": "START → llm_node → [조건분기] → tool_node → llm_node(반복) or END",
                "graph_edges": [
                    "START → llm_node",
                    "llm_node → (tool_calls?) → tool_node or END",
                    "tool_node → llm_node",
                ],
            })

            # ── 실행 ─────────────────────────────────────
            final_answer = ""
            all_states = []

            async for update in compiled.astream(
                {"messages": [HumanMessage(content=question)]},
                config={"configurable": {"thread_id": "demo-thread"}},
                stream_mode="updates",
            ):
                # 이번 스텝 실행 중 쌓인 세부 이벤트를 먼저 순서대로 전송
                for ev in pending_events:
                    yield ev
                pending_events.clear()

                for node_name, node_state in update.items():
                    all_states.append({"node": node_name, "step_count": node_state.get("step_count")})
                    if node_state.get("final_output"):
                        final_answer = node_state["final_output"]

            yield sse({"type": "arch", "step": "출력", "desc": "최종 State에서 final_output 추출"})
            yield sse({"type": "answer", "content": final_answer, "all_states": all_states})
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ════════════════════════════════════════════════
#  정적 파일(index.html) 서빙
#  - Express의 app.use(express.static("public"))와 동일한 효과
#  - public/index.html 을 "/" 에서 자동으로 서빙 (html=True)
#  - 반드시 API 라우트(@app.get(...))들을 다 등록한 "뒤"에 mount 해야
#    "/api/..." 요청이 static 핸들러에 가로채이지 않습니다.
# ════════════════════════════════════════════════
if os.path.isdir("public"):
    app.mount("/", StaticFiles(directory="public", html=True), name="public")
else:
    print("⚠️  'public' 폴더가 없습니다. public/index.html 을 넣어주세요.")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 3000))
    print(f"✅ http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
