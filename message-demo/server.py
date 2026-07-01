import os
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

load_dotenv()  # .env 파일에서 OPENAI_API_KEY 등을 로드

app = FastAPI(title="LangChain Message 클래스 실습 (Python)")


# ─────────────────────────────────────────────
# 요청 바디 스키마 (express의 req.body 역할)
# ─────────────────────────────────────────────
class Demo2Request(BaseModel):
    imageBase64: str
    mimeType: str = "image/jpeg"


class Demo3Request(BaseModel):
    question: str = "17 곱하기 42는 얼마야?"


class HistoryItem(BaseModel):
    role: str  # "human" | "ai"
    content: str


class Demo4Request(BaseModel):
    question: str
    history: list[HistoryItem] = []


def err_response(e: Exception, status_code: int = 500) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": str(e)},
    )


# ─────────────────────────────────────────────
# Demo 1: AIMessage 속성 이해
#   SystemMessage + HumanMessage → model.invoke()
#   → AIMessage { content, tool_calls, usage_metadata, response_metadata }
# ─────────────────────────────────────────────
@app.post("/api/demo1")
async def demo1():
    try:
        messages = [
            SystemMessage(content="당신은 친절한 한국어 수학 선생님입니다."),
            HumanMessage(content="15 × 23 + 7는 얼마야?"),
        ]

        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        response: AIMessage = await model.ainvoke(messages)

        meta = response.response_metadata or {}

        return {
            "success": True,
            # AIMessage 핵심 속성들
            # model_name : OpenAI 내부 모델 버전 식별자(JS의 system_fingerprint와 유사)
            # finish_reason : 모델이 왜 응답 생성을 종료했는가
            "content": response.content,
            "tool_calls": response.tool_calls if response.tool_calls else None,
            "usage_metadata": response.usage_metadata,
            "response_metadata": {
                "model": meta.get("model_name"),
                "finish_reason": meta.get("finish_reason"),
                "system_fingerprint": meta.get("system_fingerprint"),
            },
            # 전송한 메시지 클래스 정보
            "message_classes": [
                {
                    "class": m.__class__.__name__,
                    "type": m.type,
                    "content": m.content,
                }
                for m in messages
            ],
        }
    except Exception as e:
        return err_response(e)


# ─────────────────────────────────────────────
# Demo 2: HumanMessage 멀티모달
#   content: [ { type:"text" }, { type:"image_url", image_url:{url} } ]
# ─────────────────────────────────────────────
@app.post("/api/demo2")
async def demo2(body: Demo2Request):
    try:
        if not body.imageBase64:
            raise HTTPException(status_code=400, detail="이미지 데이터가 없습니다.")

        model = ChatOpenAI(model="gpt-4o", temperature=0)

        message = HumanMessage(
            content=[
                {"type": "text", "text": "이 이미지에 무엇이 있나요? 한국어로 자세히 설명해주세요."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{body.mimeType};base64,{body.imageBase64}"
                    },
                },
            ]
        )

        response: AIMessage = await model.ainvoke([message])

        content_blocks = []
        for b in message.content:
            if isinstance(b, dict) and b.get("type") == "image_url":
                kb = round(len(body.imageBase64) / 1024)
                content_blocks.append(
                    {"type": "image_url", "url": f"data:{body.mimeType};base64,[{kb}KB]"}
                )
            else:
                content_blocks.append(b)

        return {
            "success": True,
            "content": response.content,
            "message_structure": {
                "class": message.__class__.__name__,
                "type": message.type,
                "content_blocks": content_blocks,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        return err_response(e)


# ─────────────────────────────────────────────
# Demo 3: ToolMessage — Tool Calling 전체 흐름
#   HumanMessage → AIMessage(tool_calls) → ToolMessage(결과) → AIMessage(최종)
# ─────────────────────────────────────────────
class MultiplySchema(BaseModel):
    """두 숫자를 곱합니다."""

    a: float = Field(description="첫 번째 숫자")
    b: float = Field(description="두 번째 숫자")


class AddSchema(BaseModel):
    """두 숫자를 더합니다."""

    a: float = Field(description="첫 번째 숫자")
    b: float = Field(description="두 번째 숫자")


@app.post("/api/demo3")
async def demo3(body: Demo3Request):
    try:
        question = body.question or "17 곱하기 42는 얼마야?"

        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # bind_tools에 전달하는 Pydantic 클래스명이 곧 tool 이름이 됩니다.
        # JS 버전의 name:"multiply"/"add"와 동일하게 맞추기 위해
        # tool_choice 없이 두 개의 스키마를 그대로 바인딩합니다.
        model_with_tools = model.bind_tools(
            [
                {
                    "name": "multiply",
                    "description": MultiplySchema.__doc__,
                    "parameters": MultiplySchema.model_json_schema(),
                },
                {
                    "name": "add",
                    "description": AddSchema.__doc__,
                    "parameters": AddSchema.model_json_schema(),
                },
            ]
        )

        # Step 1: 첫 번째 LLM 호출 (tool_calls 포함 AIMessage 반환)
        initial_messages = [HumanMessage(content=question)]
        ai_response: AIMessage = await model_with_tools.ainvoke(initial_messages)

        # Step 2: Tool 직접 실행 + ToolMessage 생성
        tool_results = []
        tool_messages = []

        for tc in ai_response.tool_calls or []:
            a, b = tc["args"]["a"], tc["args"]["b"]
            result = a * b if tc["name"] == "multiply" else a + b
            tool_results.append({"tool": tc["name"], "args": tc["args"], "result": result})
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

        # Step 3: 전체 메시지 이력 + ToolMessage → 최종 AIMessage
        final_answer: Optional[str] = None
        if tool_messages:
            final_messages = [*initial_messages, ai_response, *tool_messages]
            final_response: AIMessage = await model_with_tools.ainvoke(final_messages)
            final_answer = final_response.content

        message_flow: list[dict[str, Any]] = [
            {"class": "HumanMessage", "content": question, "detail": None},
            {
                "class": "AIMessage (tool_calls)",
                "tool_calls": ai_response.tool_calls,
                "detail": "tool_calls 포함",
            },
            *[
                {
                    "class": "ToolMessage",
                    "tool": t["tool"],
                    "args": t["args"],
                    "result": t["result"],
                    "detail": "tool_call_id로 AIMessage와 연결",
                }
                for t in tool_results
            ],
            {"class": "AIMessage (final)", "content": final_answer, "detail": "최종 자연어 답변"},
        ]

        return {
            "success": True,
            "question": question,
            "ai_first_response": {
                "class": "AIMessage",
                "content": ai_response.content,
                "tool_calls": ai_response.tool_calls,
            },
            "tool_executions": tool_results,
            "final_answer": final_answer,
            "message_flow": message_flow,
        }
    except Exception as e:
        return err_response(e)


# ─────────────────────────────────────────────
# Demo 4: MessagesPlaceholder — 대화 이력 동적 삽입
#   [System, Few-shot Human, Few-shot AI, ...history, Human(질문)]
# ─────────────────────────────────────────────
@app.post("/api/demo4")
async def demo4(body: Demo4Request):
    try:
        question = body.question
        if not question:
            raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

        # 이력 메시지 복원
        chat_history = [
            HumanMessage(content=m.content) if m.role == "human" else AIMessage(content=m.content)
            for m in body.history
        ]

        # MessagesPlaceholder가 삽입될 위치에 chat_history를 직접 스프레드
        # (실제 ChatPromptTemplate 사용 시 MessagesPlaceholder("chat_history")로 대체)
        messages = [
            SystemMessage(
                content="너는 세계 최고의 한국어 선생님이야.\n항상 존댓말을 사용하고, 3문장 이내로 답변해."
            ),
            # Few-shot 예시
            HumanMessage(content="자음 탈락이 뭐야?"),
            AIMessage(content="말할 때 'ㄱ'이나 'ㅂ' 같은 자음이 탈락하는 현상이에요. 예: '먹고' → '멱고'"),
            # ↓ MessagesPlaceholder("chat_history") 가 런타임에 이 위치에 삽입됨
            *chat_history,
            HumanMessage(content=question),
        ]

        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        response: AIMessage = await model.ainvoke(messages)

        def short(content: Any) -> str:
            if isinstance(content, str):
                return content[:80] + "..." if len(content) > 80 else content
            return "[복합 컨텐츠]"

        return {
            "success": True,
            "answer": response.content,
            "message_chain": [
                {"class": m.__class__.__name__, "type": m.type, "content": short(m.content)}
                for m in messages
            ],
            "placeholder_explained": {
                "name": "MessagesPlaceholder",
                "variable": "chat_history",
                "injected_count": len(chat_history),
                "description": "MessagesPlaceholder는 런타임에 대화 이력을 프롬프트에 동적으로 삽입하는 템플릿 변수입니다.",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        return err_response(e)


# ─────────────────────────────────────────────
# 정적 파일 서빙 (express.static("public") 대응)
# ※ API 라우트들을 먼저 등록한 뒤 마지막에 마운트해야
#    /api/* 경로가 정적 파일 매칭에 가로채이지 않습니다.
# ─────────────────────────────────────────────
app.mount("/", StaticFiles(directory="public", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 3000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
