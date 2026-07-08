"""
server.py
=========
server.js(Node.js/Express, "4개 커스텀 미들웨어 + createReactAgent" 데모)를
Python(FastAPI)으로 이식한 버전입니다.

포함된 4개 미들웨어 (모두 Agent 실행 파이프라인에 직접 끼워 넣는 방식으로 구현)
  1) model_call_limit_middleware — LLM 호출 횟수 제한
  2) tool_call_limit_middleware  — Tool 호출 횟수 제한
  3) summarization_middleware    — 메시지가 일정 개수 쌓이면 자동 요약
  4) human_in_the_loop_middleware — Tool 실행 전 사람 승인 대기 (최대 30초 폴링)

엔드포인트 (원본 Express 서버와 동일하게 유지 — index.html 수정 없이 그대로 사용 가능)
  GET  /api/agent/stream        SSE로 Agent 실행 스트리밍
  POST /api/session/config      미들웨어 설정 변경
  GET  /api/session/status      세션 상태 조회 (승인 대기 여부 포함)
  POST /api/session/approve     Human-in-the-loop 승인/거절
  POST /api/session/reset       세션 초기화

실행 방법
---------
    pip install fastapi uvicorn langchain-openai langgraph tavily-python python-dotenv
    # .env 에 OPENAI_API_KEY=sk-... (선택: TAVILY_API_KEY=tvly-...)
    python server.py
"""

import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()

app = FastAPI()


# ════════════════════════════════════════════════
#  뉴스 검색 (Tavily 있으면 실제 검색, 없으면 GPT로 시뮬레이션)
# ════════════════════════════════════════════════
async def news_search_impl(query: str) -> str:
    tavily_key = os.environ.get("TAVILY_API_KEY")

    if tavily_key and tavily_key != "your-tavily-key":
        try:
            from tavily import AsyncTavilyClient

            client = AsyncTavilyClient(api_key=tavily_key)
            result = await client.search(query, max_results=5, search_depth="advanced")
            return "\n\n".join(
                f"{i + 1}. {r.get('title')}\n   {r.get('url')}\n   {(r.get('content') or '')[:200]}"
                for i, r in enumerate(result.get("results", []))
            )
        except Exception as e:
            return f"Tavily 오류: {e}"

    # fallback: GPT로 뉴스 시뮬레이션
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
    res = await model.ainvoke(
        f'"{query}" 관련 최신 뉴스를 5개 작성해줘. '
        f"제목, 출처, 핵심 내용(2문장) 형식으로 번호 붙여서. 최신 트렌드 반영."
    )
    return res.content


# ════════════════════════════════════════════════
#  세션 저장소 (미들웨어 상태 추적)
# ════════════════════════════════════════════════
sessions: dict[str, dict[str, Any]] = {}  # sessionId -> 세션 데이터


def get_session(session_id: str) -> dict[str, Any]:
    if session_id not in sessions:
        sessions[session_id] = {
            "id": session_id,
            "logs": [],  # 미들웨어 로그
            "model_call_count": 0,  # model_call_limit_middleware용
            "tool_call_count": 0,  # tool_call_limit_middleware용
            "summaries": [],  # summarization_middleware용
            "pending_approval": None,  # human_in_the_loop_middleware용
            "messages": [],  # 전체 메시지 이력
            "config": {
                "modelRunLimit": 5,
                "toolRunLimit": 3,
                "summarizeThreshold": 4,  # 메시지 N개 넘으면 요약
                "requireApproval": False,  # human-in-the-loop 활성화 여부
            },
        }
    return sessions[session_id]


def add_log(session: dict, middleware: str, level: str, message: str, detail: Optional[str] = None) -> dict:
    """level: 'info' | 'warn' | 'block' | 'pass'"""
    log = {
        "id": f"{time.time()}-{uuid.uuid4().hex[:6]}",
        "middleware": middleware,
        "level": level,
        "message": message,
        "detail": detail,
        "ts": datetime.now().strftime("%p %I:%M:%S").replace("AM", "오전").replace("PM", "오후"),
    }
    session["logs"].append(log)
    return log


# ════════════════════════════════════════════════
#  미들웨어 구현
# ════════════════════════════════════════════════

# ── Middleware 1: model_call_limit_middleware ─────
# LLM 호출 횟수 제한
async def model_call_limit_middleware(session: dict) -> None:
    session["model_call_count"] += 1
    limit = session["config"]["modelRunLimit"]
    if session["model_call_count"] > limit:
        add_log(
            session, "modelCallLimit", "block",
            f"LLM 호출 한도 초과 ({session['model_call_count']}/{limit})",
            f"설정된 최대 호출 횟수({limit})를 초과해 실행을 중단합니다.",
        )
        raise RuntimeError(f"[modelCallLimit] LLM 호출 한도({limit}회)를 초과했습니다.")
    add_log(
        session, "modelCallLimit", "pass",
        f"LLM 호출 허용 ({session['model_call_count']}/{limit})",
        f"남은 호출 가능 횟수: {limit - session['model_call_count']}회",
    )


# ── Middleware 2: tool_call_limit_middleware ──────
# Tool 호출 횟수 제한
async def tool_call_limit_middleware(session: dict) -> None:
    session["tool_call_count"] += 1
    limit = session["config"]["toolRunLimit"]
    if session["tool_call_count"] > limit:
        add_log(
            session, "toolCallLimit", "block",
            f"Tool 호출 한도 초과 ({session['tool_call_count']}/{limit})",
            "도구 실행 횟수가 한도를 초과했습니다.",
        )
        raise RuntimeError(f"[toolCallLimit] Tool 호출 한도({limit}회)를 초과했습니다.")
    add_log(
        session, "toolCallLimit", "pass",
        f"Tool 호출 허용 ({session['tool_call_count']}/{limit})",
        f"남은 Tool 호출 가능 횟수: {limit - session['tool_call_count']}회",
    )


# ── Middleware 3: summarization_middleware ──────
# 대화 이력이 길어지면 요약해서 컨텍스트 압축
async def summarization_middleware(session: dict) -> Optional[str]:
    threshold = session["config"]["summarizeThreshold"]
    msg_count = len(session["messages"])

    if msg_count > 0 and msg_count % threshold == 0:
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        recent = session["messages"][-threshold:]
        to_summarize = "\n".join(
            f"[{m['role']}]: {(m['content'] if isinstance(m['content'], str) else json.dumps(m['content'], ensure_ascii=False))[:300]}"
            for m in recent
        )

        summary = await model.ainvoke(f"다음 대화를 3문장 이내로 핵심만 요약해줘:\n\n{to_summarize}")
        summary_text = summary.content
        session["summaries"].append({"at": msg_count, "text": summary_text})

        add_log(
            session, "summarization", "info",
            f"대화 요약 실행 (메시지 {msg_count}개 → 요약본 생성)",
            f"요약: {summary_text}",
        )
        return summary_text

    remainder = threshold - (msg_count % threshold or threshold)
    add_log(
        session, "summarization", "pass",
        f"요약 불필요 (현재 {msg_count}개 / 기준 {threshold}개)",
        f"{remainder}개 더 쌓이면 자동 요약됩니다.",
    )
    return None


# ── Middleware 4: human_in_the_loop_middleware ─────
# Tool 실행 전 사람의 승인 요청 (최대 30초 폴링 대기)
async def human_in_the_loop_middleware(session: dict, tool_name: str, tool_args: dict) -> dict:
    import asyncio

    if not session["config"]["requireApproval"]:
        add_log(
            session, "humanInTheLoop", "pass",
            "자동 승인 (Human-in-the-Loop 비활성화)",
            f'requireApproval=false: 도구 "{tool_name}" 자동 실행',
        )
        return {"approved": True, "auto": True}

    # 승인 대기 상태로 설정 (별도 REST 엔드포인트 /api/session/status 로 프론트엔드가 폴링해서 확인)
    session["pending_approval"] = {
        "toolName": tool_name,
        "toolArgs": tool_args,
        "status": "pending",
        "requestedAt": datetime.now().isoformat(),
    }

    add_log(
        session, "humanInTheLoop", "warn",
        f'사람 승인 대기 중 — Tool: "{tool_name}"',
        f"실행 인자: {json.dumps(tool_args, ensure_ascii=False)}\n승인/거절 버튼을 클릭하세요.",
    )

    # 최대 30초 대기 (200ms 간격으로 상태 확인)
    timeout_sec, interval_sec, elapsed = 30.0, 0.2, 0.0
    while elapsed < timeout_sec:
        await asyncio.sleep(interval_sec)
        elapsed += interval_sec
        if session["pending_approval"] is None or session["pending_approval"].get("status") != "pending":
            break

    result = session["pending_approval"]
    session["pending_approval"] = None

    if result and result.get("status") == "approved":
        add_log(session, "humanInTheLoop", "info", f'✅ 사람이 승인함 — Tool: "{tool_name}"', "도구 실행을 허가합니다.")
        return {"approved": True, "auto": False}

    add_log(session, "humanInTheLoop", "block", f'❌ 사람이 거절함 — Tool: "{tool_name}"', "도구 실행이 거부되었습니다.")
    return {"approved": False, "auto": False}


def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ════════════════════════════════════════════════
#  미들웨어 래핑된 Agent 실행 (SSE 스트리밍)
# ════════════════════════════════════════════════
@app.get("/api/agent/stream")
async def agent_stream(question: Optional[str] = Query(None), sessionId: str = Query("default")):
    if not question:
        return JSONResponse(status_code=400, content={"error": "question 필요"})

    session = get_session(sessionId)

    async def event_generator():
        try:
            yield sse({"type": "start", "message": "Agent 시작", "sessionId": sessionId})

            # ── Middleware 3: Summarization (사전 체크) ──
            summary = await summarization_middleware(session)
            yield sse({"type": "middleware", "name": "summarizationMiddleware", "log": session["logs"][-1]})

            system_context = (
                f"[이전 대화 요약]\n{summary}\n\n당신은 뉴스 검색 AI 어시스턴트입니다."
                if summary
                else "당신은 뉴스 검색 AI 어시스턴트입니다. 사용자 질문에 news_search 도구를 사용해 최신 뉴스를 검색하세요."
            )

            # ── Middleware 1: modelCallLimit ─────────────
            try:
                await model_call_limit_middleware(session)
                yield sse({"type": "middleware", "name": "modelCallLimitMiddleware", "log": session["logs"][-1]})
            except RuntimeError as e:
                yield sse({"type": "middleware", "name": "modelCallLimitMiddleware", "log": session["logs"][-1]})
                yield sse({"type": "blocked", "message": str(e)})
                yield sse({"type": "done"})
                return

            # 노드 실행 도중 만든 SSE 이벤트를 잠시 담아두는 큐 (Tool 실행 전체가 끝나야 다음 astream 결과가 나오므로,
            # Tool 안에서 발생한 세부 이벤트들은 여기 모았다가 그 결과와 함께 순서대로 내보냅니다)
            pending_events: list[str] = []

            # ── Tool 호출 전 미들웨어 적용을 위한 래퍼 Tool ─
            @tool
            async def news_search(query: str) -> str:
                """최신 뉴스를 검색합니다. AI, 기술, 경제 등 최신 정보가 필요할 때 사용합니다.

                Args:
                    query: 검색할 뉴스 키워드
                """
                # ── Middleware 2: toolCallLimit ────────────
                try:
                    await tool_call_limit_middleware(session)
                    pending_events.append(sse({"type": "middleware", "name": "toolCallLimitMiddleware", "log": session["logs"][-1]}))
                except RuntimeError as e:
                    pending_events.append(sse({"type": "middleware", "name": "toolCallLimitMiddleware", "log": session["logs"][-1]}))
                    return f"[toolCallLimit 차단] {e}"

                # ── Middleware 4: humanInTheLoop ───────────
                approval = await human_in_the_loop_middleware(session, "news_search", {"query": query})
                pending_events.append(sse({"type": "middleware", "name": "humanInTheLoopMiddleware", "log": session["logs"][-1]}))

                if not approval["approved"]:
                    return "[Human-in-the-Loop 거절] 사용자가 뉴스 검색을 거부했습니다."

                # 실제 뉴스 검색 실행
                pending_events.append(sse({"type": "tool_call", "tool": "news_search", "args": {"query": query}}))
                result = await news_search_impl(query)
                pending_events.append(sse({"type": "tool_result", "tool": "news_search", "content": result[:500]}))
                return result

            # Agent 생성 (MemorySaver로 대화 이력 유지)
            checkpointer = MemorySaver()
            model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

            agent = create_react_agent(
                model=model,
                tools=[news_search],
                checkpointer=checkpointer,
                prompt=SystemMessage(content=system_context),
            )

            # Agent 스트리밍 실행
            final_answer = ""

            async for update in agent.astream(
                {"messages": [HumanMessage(content=question)]},
                config={"configurable": {"thread_id": sessionId}},
                stream_mode="updates",
            ):
                # Tool 실행 중 쌓인 이벤트를 먼저 순서대로 전송
                for ev in pending_events:
                    yield ev
                pending_events.clear()

                if "agent" in update:
                    for msg in update["agent"].get("messages", []):
                        if getattr(msg, "tool_calls", None):
                            for tc in msg.tool_calls:
                                yield sse({"type": "agent_thinking", "tool": tc["name"], "args": tc["args"]})
                        if isinstance(msg.content, str) and msg.content.strip():
                            final_answer = msg.content
                            yield sse({"type": "answer", "content": msg.content})

            # 메시지 이력 업데이트
            session["messages"].append({"role": "user", "content": question})
            session["messages"].append({"role": "assistant", "content": final_answer})

            yield sse({
                "type": "done",
                "stats": {
                    "modelCalls": session["model_call_count"],
                    "toolCalls": session["tool_call_count"],
                    "summaries": len(session["summaries"]),
                    "messages": len(session["messages"]),
                },
            })
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── 세션 설정 변경 ────────────────────────────
class SessionConfigRequest(BaseModel):
    sessionId: str = "default"
    config: dict[str, Any]


@app.post("/api/session/config")
async def update_session_config(body: SessionConfigRequest):
    session = get_session(body.sessionId)
    session["config"].update(body.config)
    return {"success": True, "config": session["config"]}


# ── 세션 상태 조회 ────────────────────────────
@app.get("/api/session/status")
async def session_status(sessionId: str = Query("default")):
    session = get_session(sessionId)
    return {
        "success": True,
        "config": session["config"],
        "modelCallCount": session["model_call_count"],
        "toolCallCount": session["tool_call_count"],
        "messageCount": len(session["messages"]),
        "summaryCount": len(session["summaries"]),
        "summaries": session["summaries"],
        "pendingApproval": session["pending_approval"],
        "recentLogs": session["logs"][-20:],
    }


# ── Human-in-the-Loop 승인/거절 ───────────────
class ApproveRequest(BaseModel):
    sessionId: str = "default"
    approved: bool


@app.post("/api/session/approve")
async def approve_session(body: ApproveRequest):
    session = get_session(body.sessionId)
    if session["pending_approval"]:
        session["pending_approval"]["status"] = "approved" if body.approved else "rejected"
        return {"success": True, "status": session["pending_approval"]["status"]}
    return {"success": False, "error": "대기 중인 승인 요청이 없습니다."}


# ── 세션 초기화 ───────────────────────────────
class ResetRequest(BaseModel):
    sessionId: str = "default"


@app.post("/api/session/reset")
async def reset_session(body: ResetRequest):
    sessions.pop(body.sessionId, None)
    return {"success": True}


# ════════════════════════════════════════════════
#  정적 파일(index.html) 서빙
#  - Express의 app.use(express.static("public"))와 동일한 효과
#  - 반드시 API 라우트들을 다 등록한 "뒤"에 mount 해야
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
