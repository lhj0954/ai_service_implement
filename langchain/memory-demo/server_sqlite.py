"""
1. InMemoryChatMessageHistory  — 프로세스 메모리 저장, 재시작 시 소멸
2. SimpleFileChatMessageHistory — JSON 파일 저장, 재시작해도 유지
3. SimpleSQLiteChatMessageHistory — SQLite DB 파일 저장, 재시작해도 유지

"""

import json
import os
from pathlib import Path
from typing import List, Optional

import sqlite3
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
#  3. SimpleSQLiteChatMessageHistory
#  — SQLite DB 파일로 저장, 서버 재시작에도 유지됨
# ════════════════════════════════════════════════
class SqliteChatRequest(ChatRequest):
    dbPath: str = "chat_histories/chat_history.db"


class SqliteResetRequest(ResetRequest):
    dbPath: str = "chat_histories/chat_history.db"


class SimpleSQLiteChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, db_path: str = "chat_histories/chat_history.db"):
        self.session_id = session_id
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id)"
            )
            conn.commit()

    @property
    def messages(self) -> List[BaseMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (self.session_id,),
            ).fetchall()

        result: List[BaseMessage] = []
        for role, content in rows:
            if role == "human":
                result.append(HumanMessage(content=content))
            else:
                result.append(AIMessage(content=content))
        return result

    def add_message(self, message: BaseMessage) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (session_id, role, content)
                VALUES (?, ?, ?)
                """,
                (self.session_id, message.type, message.content),
            )
            conn.commit()

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (self.session_id,))
            conn.commit()

    def raw_messages(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (self.session_id,),
            ).fetchall()
        return [
            {"id": row[0], "role": row[1], "content": row[2], "created_at": row[3]}
            for row in rows
        ]


sqlite_histories: dict[str, SimpleSQLiteChatMessageHistory] = {}


# 같은 세션과 DB 경로는 동일한 history 객체를 공유
# Redis와 달리 별도 서버 실행이 필요 없고, dbPath 위치에 .db 파일이 생성됨
def get_sqlite_history(session_id: str, db_path: str) -> SimpleSQLiteChatMessageHistory:
    key = f"{session_id}:{db_path}"
    if key not in sqlite_histories:
        sqlite_histories[key] = SimpleSQLiteChatMessageHistory(session_id, db_path)
    return sqlite_histories[key]


@app.post("/api/sqlite/chat")
async def sqlite_chat(body: SqliteChatRequest):
    if not body.input:
        return JSONResponse(status_code=400, content={"success": False, "error": "input 필요"})

    try:
        history = get_sqlite_history(body.sessionId, body.dbPath)

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

        return {
            "success": True,
            "answer": response.content,
            "history": msgs_to_json(history.messages),
            "backend": "SimpleSQLiteChatMessageHistory",
            "dbPath": str(history.db_path),
            "table": "chat_messages",
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/sqlite/reset")
async def sqlite_reset(body: SqliteResetRequest):
    try:
        history = get_sqlite_history(body.sessionId, body.dbPath)
        history.clear()
        sqlite_histories.pop(f"{body.sessionId}:{body.dbPath}", None)
        return {"success": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": "SQLite 초기화 실패: " + str(e)})


@app.get("/api/sqlite/raw")
async def sqlite_raw(
    sessionId: str = Query(default="demo"),
    dbPath: str = Query(default="chat_histories/chat_history.db"),
):
    try:
        history = get_sqlite_history(sessionId, dbPath)
        return {
            "success": True,
            "raw": history.raw_messages(),
            "dbPath": str(history.db_path),
            "table": "chat_messages",
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/api/sqlite/status")
async def sqlite_status(dbPath: str = Query(default="chat_histories/chat_history.db")):
    try:
        db_path = Path(dbPath)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            conn.execute("SELECT 1")
        return {"success": True, "connected": True, "dbPath": str(db_path)}
    except Exception as e:
        return {"success": True, "connected": False, "dbPath": dbPath, "error": str(e)}


# 기존 프론트엔드가 /api/redis/...를 호출하고 있다면 바로 깨지지 않도록 만든 호환용 엔드포인트
# 내부 저장소는 Redis가 아니라 SQLite를 사용함
@app.post("/api/redis/chat")
async def redis_chat_compat(body: SqliteChatRequest):
    return await sqlite_chat(body)


@app.post("/api/redis/reset")
async def redis_reset_compat(body: SqliteResetRequest):
    return await sqlite_reset(body)


@app.get("/api/redis/status")
async def redis_status_compat(dbPath: str = Query(default="chat_histories/chat_history.db")):
    return await sqlite_status(dbPath)


# express.static("public")과 동일 — 반드시 API 라우트 아래에 마운트
app.mount("/", StaticFiles(directory="public", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 3000))
    print(f"✅ http://localhost:{port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
