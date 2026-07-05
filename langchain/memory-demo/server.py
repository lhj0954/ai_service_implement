import json
import os
from pathlib import Path
from typing import List, Optional

import redis as redis_lib
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

app = FastAPI(title="LangChain Chat Memory 백엔드 비교 (FastAPI)")


# ── 공통 프롬프트 (chat_history placeholder 필수) ──────────
def build_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "너는 친절한 한국어 선생님이야. 이전 대화를 기억하며 3문장 이내로 답변해."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    return prompt | model


# ── 요청 바디 모델 ────────────────────────────────────────
class ChatRequest(BaseModel):
    input: Optional[str] = None
    sessionId: str = "demo"


class RedisChatRequest(ChatRequest):
    redisUrl: str = "redis://localhost:6379"


class ResetRequest(BaseModel):
    sessionId: str = "demo"


class RedisResetRequest(ResetRequest):
    redisUrl: str = "redis://localhost:6379"


def msgs_to_json(messages: List[BaseMessage]) -> list:
    """BaseMessage 리스트를 프론트엔드가 기대하는 {role, content} 형태로 변환"""
    return [{"role": m.type, "content": m.content} for m in messages]


# ════════════════════════════════════════════════
#  1. InMemoryChatMessageHistory
#  — 프로세스 메모리에 저장, 서버 재시작 시 사라짐
# ════════════════════════════════════════════════
memory_stores: dict[str, InMemoryChatMessageHistory] = {}  # sessionId -> InMemoryChatMessageHistory


def get_memory_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in memory_stores:
        memory_stores[session_id] = InMemoryChatMessageHistory()
    return memory_stores[session_id]


memory_chain = RunnableWithMessageHistory(
    build_chain(),
    get_session_history=get_memory_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


@app.post("/api/memory/chat")
async def memory_chat(body: ChatRequest):
    if not body.input:
        return JSONResponse(status_code=400, content={"success": False, "error": "input 필요"})
    try:
        response = await memory_chain.ainvoke(
            {"input": body.input},
            config={"configurable": {"session_id": body.sessionId}},
        )
        history = get_memory_history(body.sessionId)
        return {
            "success": True,
            "answer": response.content,
            "history": msgs_to_json(history.messages),
            "backend": "InMemoryChatMessageHistory",
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/memory/reset")
async def memory_reset(body: ResetRequest):
    memory_stores.pop(body.sessionId, None)
    return {"success": True}


# ════════════════════════════════════════════════
#  2. SimpleFileChatMessageHistory
#  — JSON 파일로 저장, 서버 재시작에도 유지됨
# ════════════════════════════════════════════════
class SimpleFileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.file_path = Path.cwd() / "chat_histories" / f"{session_id}.json"

    @property
    def messages(self) -> List[BaseMessage]:
        try:
            raw = json.loads(self.file_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        return [
            HumanMessage(content=m["content"]) if m["role"] == "human" else AIMessage(content=m["content"])
            for m in raw
        ]

    def add_message(self, message: BaseMessage) -> None:
        messages = self.messages
        messages.append(message)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = [{"role": m.type, "content": m.content} for m in messages]
        self.file_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear(self) -> None:
        self.file_path.unlink(missing_ok=True)


file_histories: dict[str, SimpleFileChatMessageHistory] = {}


def get_file_history(session_id: str) -> SimpleFileChatMessageHistory:
    if session_id not in file_histories:
        file_histories[session_id] = SimpleFileChatMessageHistory(session_id)
    return file_histories[session_id]


file_chain = RunnableWithMessageHistory(
    build_chain(),
    get_session_history=get_file_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


@app.post("/api/file/chat")
async def file_chat(body: ChatRequest):
    if not body.input:
        return JSONResponse(status_code=400, content={"success": False, "error": "input 필요"})
    try:
        response = await file_chain.ainvoke(
            {"input": body.input},
            config={"configurable": {"session_id": body.sessionId}},
        )
        history = get_file_history(body.sessionId)
        file_path = f"chat_histories/{body.sessionId}.json"
        return {
            "success": True,
            "answer": response.content,
            "history": msgs_to_json(history.messages),
            "backend": "SimpleFileChatMessageHistory",
            "filePath": file_path,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/file/reset")
async def file_reset(body: ResetRequest):
    history = get_file_history(body.sessionId)
    history.clear()
    file_histories.pop(body.sessionId, None)
    return {"success": True}


@app.get("/api/file/raw")
async def file_raw(sessionId: str = Query(default="demo")):
    file_path = Path.cwd() / "chat_histories" / f"{sessionId}.json"
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raw = []
    return {"success": True, "raw": raw, "filePath": f"chat_histories/{sessionId}.json"}


# ════════════════════════════════════════════════
#  3. RedisChatMessageHistory (직접 구현)
#  — Redis 서버에 저장, TTL 지원, 여러 서버 간 공유 가능
# ════════════════════════════════════════════════
class SimpleRedisChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, url: str, ttl: Optional[int] = None):
        self.session_id = session_id
        self.key = f"chat_history:{session_id}"
        self.ttl = ttl
        self.client = redis_lib.Redis.from_url(
            url,
            socket_connect_timeout=2,
            socket_timeout=2,
            retry_on_timeout=False,
        )

    @property
    def messages(self) -> List[BaseMessage]:
        raw = self.client.lrange(self.key, 0, -1)  # key에 해당하는 List에서 모든 요소를 0번 인덱스부터 끝까지 가져옴
        result = []
        for item in raw:
            m = json.loads(item)
            result.append(
                HumanMessage(content=m["content"]) if m["role"] == "human" else AIMessage(content=m["content"])
            )
        return result

    def add_message(self, message: BaseMessage) -> None:
        serialized = json.dumps({"role": message.type, "content": message.content})
        self.client.rpush(self.key, serialized)
        if self.ttl:
            self.client.expire(self.key, self.ttl)

    def clear(self) -> None:
        self.client.delete(self.key)

    def disconnect(self) -> None:
        self.client.close()


redis_histories: dict[str, SimpleRedisChatMessageHistory] = {}


# 같은 세션은 동일한 history 객체를 공유
def get_redis_history(session_id: str, redis_url: str) -> SimpleRedisChatMessageHistory:
    key = f"{session_id}:{redis_url}"
    if key not in redis_histories:
        # Redis에 저장된 채팅 기록이 1시간 후 자동 만료
        redis_histories[key] = SimpleRedisChatMessageHistory(session_id, redis_url, ttl=3600)
    return redis_histories[key]


@app.post("/api/redis/chat")
async def redis_chat(body: RedisChatRequest):
    if not body.input:
        return JSONResponse(status_code=400, content={"success": False, "error": "input 필요"})

    history = get_redis_history(body.sessionId, body.redisUrl)

    # 연결 테스트 (redis-py의 socket_connect_timeout/socket_timeout으로 무한 대기 방지)
    try:
        history.client.ping()
    except Exception as e:
        history.client.close()
        redis_histories.pop(f"{body.sessionId}:{body.redisUrl}", None)
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": f"Redis 서버에 연결할 수 없습니다 ({body.redisUrl}). Redis가 실행 중인지 확인하세요.",
                "detail": str(e),
                "backend": "RedisChatMessageHistory",
                "redisUrl": body.redisUrl,
            },
        )

    try:
        chain = RunnableWithMessageHistory(
            build_chain(),
            get_session_history=lambda session_id: history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        response = await chain.ainvoke(
            {"input": body.input},
            config={"configurable": {"session_id": body.sessionId}},
        )
        ttl = history.client.ttl(history.key)

        return {
            "success": True,
            "answer": response.content,
            "history": msgs_to_json(history.messages),
            "backend": "RedisChatMessageHistory",
            "redisKey": history.key,
            "ttl": ttl,
            "redisUrl": body.redisUrl,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/redis/reset")
async def redis_reset(body: RedisResetRequest):
    history = get_redis_history(body.sessionId, body.redisUrl)
    try:
        history.clear()
        return {"success": True}
    except Exception as e:
        return JSONResponse(status_code=503, content={"success": False, "error": "Redis 연결 실패: " + str(e)})


@app.get("/api/redis/status")
async def redis_status(redisUrl: str = Query(default="redis://localhost:6379")):
    test_client = redis_lib.Redis.from_url(
        redisUrl,
        socket_connect_timeout=1.5,
        socket_timeout=1.5,
        retry_on_timeout=False,
    )
    try:
        test_client.ping()
        test_client.close()
        return {"success": True, "connected": True, "redisUrl": redisUrl}
    except Exception as e:
        test_client.close()
        return {"success": True, "connected": False, "redisUrl": redisUrl, "error": str(e)}


# express.static("public")과 동일 — 반드시 API 라우트 아래에 마운트
app.mount("/", StaticFiles(directory="public", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 3000))
    print(f"✅ http://localhost:{port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
