"""
server.py
=========
server.js(Node.js/Express, PDF RAG + Agent Tools 버전)를 Python(FastAPI)으로 이식한 버전입니다.

포함 기능
  - Tool 4종: get_current_time, get_weather, get_ai_news(Tavily 선택적), search_pdf_document(RAG)
  - PDF 업로드 → 청크 분할 → 임베딩 → 인메모리 벡터스토어(SimpleVectorStore)
  - create_react_agent 기반 Agent
  - /api/upload-pdf   (POST)  : PDF 업로드 & 임베딩
  - /api/pdf-status   (GET)   : PDF 준비 상태 조회
  - /api/agent/stream (GET)   : SSE 스트리밍 실행
  - /api/agent/run    (POST)  : 일반 JSON 실행

실행 방법
---------
    pip install fastapi uvicorn python-multipart langchain-openai langgraph \
                langchain-text-splitters langchain-core pypdf tavily-python python-dotenv
    # .env 파일에 OPENAI_API_KEY=sk-... (선택: TAVILY_API_KEY=tvly-...) 설정
    python server.py
"""

import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from pypdf import PdfReader

load_dotenv()

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════
#  SimpleVectorStore (PDF RAG용)
# ════════════════════════════════════════════════
class SimpleVectorStore:
    def __init__(self):
        self.docs: list[dict] = []

    @staticmethod
    def cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb + 1e-10)

    async def add_documents(self, docs: list[Document], embedder: OpenAIEmbeddings) -> int:
        embeddings = await embedder.aembed_documents([d.page_content for d in docs])
        for doc, emb in zip(docs, embeddings):
            self.docs.append({"pageContent": doc.page_content, "metadata": doc.metadata, "embedding": emb})
        return len(self.docs)

    async def similarity_search(self, query: str, k: int, embedder: OpenAIEmbeddings) -> list[dict]:
        q_emb = await embedder.aembed_query(query)
        scored = [{**d, "score": self.cosine_sim(q_emb, d["embedding"])} for d in self.docs]
        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[:k]

    def clear(self):
        self.docs = []

    @property
    def size(self) -> int:
        return len(self.docs)


pdf_store = SimpleVectorStore()
pdf_ready = False
pdf_filename = ""


# ════════════════════════════════════════════════
#  Tool 정의
# ════════════════════════════════════════════════
from langchain_core.tools import tool  # noqa: E402  (아래 tool 함수들에서 사용)


# ── Tool 1: 현재 시간 ─────────────────────────
@tool
def get_current_time() -> str:
    """현재 날짜와 시간을 반환합니다. 시간이나 날짜 관련 질문에 사용하세요."""
    now = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y. %m. %d. %p %I:%M:%S").replace("AM", "오전").replace("PM", "오후")
    return f"현재 한국 시간: {now}"


# ── Tool 2: 날씨 조회 ─────────────────────────
# open-meteo 등 외부 날씨 API 대신 OpenAI로 현실적 날씨 생성
# (실제 프로덕션에서는 openweathermap 등 실제 날씨 API 사용을 권장)
@tool
async def get_weather(city: str) -> str:
    """특정 도시의 현재 날씨를 조회합니다. 날씨 관련 질문에 사용하세요.

    Args:
        city: 날씨를 조회할 도시명 (예: 서울, 부산, 도쿄)
    """
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    month = now.month
    season = "봄" if 3 <= month <= 5 else "여름" if 6 <= month <= 8 else "가을" if 9 <= month <= 11 else "겨울"

    response = await model.ainvoke(
        f"현재 {now.strftime('%Y-%m-%d %H:%M')} 기준, {city}의 {season} 날씨를 사실적으로 예측해줘. "
        f"기온(°C), 날씨 상태(맑음/흐림/비/눈 등), 습도(%), 바람(m/s)을 포함해서 한 문단으로 자연스럽게 답해줘. "
        f"실제 날씨 데이터처럼 구체적인 숫자를 포함해줘."
    )
    return f"[{city} 날씨 정보]\n{response.content}"


# ── Tool 3: 최신 AI 뉴스 검색 ─────────────────
@tool
async def get_ai_news(query: str = "최신 AI 뉴스") -> str:
    """최신 AI 및 인공지능 관련 뉴스를 검색합니다. AI 기술 동향, 신제품, 연구 결과 등을 알고 싶을 때 사용하세요.

    Args:
        query: 검색할 뉴스 키워드 (예: GPT-5, 클로드, 구글 AI)
    """
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    # TAVILY_API_KEY가 있으면 실제 검색, 없으면 OpenAI로 시뮬레이션
    if tavily_api_key and tavily_api_key != "your-tavily-key-here":
        try:
            from tavily import AsyncTavilyClient

            client = AsyncTavilyClient(api_key=tavily_api_key)
            result = await client.search(query or "최신 AI 인공지능 뉴스", max_results=5, search_depth="advanced")
            articles = "\n\n".join(
                f"{i + 1}. [{r.get('title')}]\n   {r.get('url')}\n   {(r.get('content') or '')[:150]}..."
                for i, r in enumerate(result.get("results", []))
            )
            return f'[최신 AI 뉴스 검색 결과]\n검색어: "{query}"\n\n{articles}'
        except Exception as e:
            return f"Tavily 검색 오류: {e}"
    else:
        # TAVILY_API_KEY 없을 때 OpenAI로 최신 트렌드 기반 뉴스 생성
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
        response = await model.ainvoke(
            f'다음 검색어에 대한 최신 AI 뉴스를 5개 작성해줘: "{query or "AI 인공지능 최신 동향"}"\n'
            f"각 뉴스는 제목, 출처(가상), 핵심 내용(2-3문장) 형식으로 번호를 붙여서 작성해줘. "
            f"최신 AI 트렌드(LLM, 에이전트, 멀티모달, 자율주행 AI 등)를 반영해줘."
        )
        return f'[AI 뉴스 (시뮬레이션 - TAVILY_API_KEY 미설정)]\n검색어: "{query}"\n\n{response.content}'


# ── Tool 4: PDF 문서 검색 (RAG) ───────────────
@tool
async def search_pdf_document(question: str) -> str:
    """업로드된 PDF 문서(회사 규정, 매뉴얼 등)에서 정보를 검색합니다. 회사 정책, 규정, 절차 등을 물어볼 때 사용하세요.

    Args:
        question: PDF 문서에서 찾고 싶은 내용 (예: 연차 규정, 복지 혜택)
    """
    if not pdf_ready or pdf_store.size == 0:
        return "PDF가 업로드되지 않았습니다. 먼저 PDF 파일을 업로드해주세요."

    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    docs = await pdf_store.similarity_search(question, 4, embedder)
    context = "\n\n".join(d["pageContent"] for d in docs)
    sources = ", ".join(f"p.{d['metadata'].get('page', '?')}" for d in docs)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    response = await model.ainvoke(
        "다음 회사 규정 문서 내용을 바탕으로 질문에 답해줘. 문서에 없는 내용은 "
        '"해당 내용이 문서에 없습니다"라고 말해.\n\n'
        f"[문서 내용]\n{context}\n\n[질문] {question}"
    )
    return f"[PDF 검색 결과 - {pdf_filename}] (참조: {sources})\n\n{response.content}"


# ════════════════════════════════════════════════
#  Agent 생성 함수
# ════════════════════════════════════════════════
def create_agent():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    tools = [get_current_time, get_weather, get_ai_news, search_pdf_document]

    return create_react_agent(
        model=model,
        tools=tools,
        prompt=(
            "당신은 유능한 AI 어시스턴트입니다. 사용자의 질문에 따라 적절한 도구를 선택해 답변합니다.\n"
            "- 시간/날짜 → get_current_time\n"
            "- 날씨 → get_weather\n"
            "- AI 뉴스/동향 → get_ai_news\n"
            "- 회사 규정/PDF 문서 → search_pdf_document\n"
            "항상 한국어로 친절하게 답변하세요."
        ),
    )


def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ════════════════════════════════════════════════
#  API Routes
# ════════════════════════════════════════════════

# PDF 업로드
@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_ready, pdf_filename
    try:
        if not file:
            return JSONResponse(status_code=400, content={"success": False, "error": "PDF 파일을 업로드하세요."})

        tmp_path = UPLOAD_DIR / file.filename
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        # 페이지별 텍스트 추출 (Node의 pdf-parse \f 페이지 구분과 동일한 효과)
        reader = PdfReader(str(tmp_path))
        pages_text = [(page.extract_text() or "") for page in reader.pages]
        pages = [p for p in pages_text if len(p.strip()) > 50]
        page_list = pages if pages else (pages_text or [""])

        docs = [
            Document(
                page_content=" ".join(text.split()),
                metadata={"source": file.filename, "page": i + 1},
            )
            for i, text in enumerate(page_list)
        ]

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        pdf_store.clear()
        embedder = OpenAIEmbeddings(model="text-embedding-3-small")
        count = await pdf_store.add_documents(chunks, embedder)
        pdf_ready = True
        pdf_filename = file.filename

        tmp_path.unlink(missing_ok=True)
        return {"success": True, "filename": pdf_filename, "page_count": len(page_list), "chunk_count": count}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# PDF 상태
@app.get("/api/pdf-status")
async def pdf_status():
    return {"ready": pdf_ready, "filename": pdf_filename, "chunk_count": pdf_store.size}


# Agent 실행 (스트리밍 - SSE)
@app.get("/api/agent/stream")
async def agent_stream(question: Optional[str] = Query(None)):
    if not question:
        return JSONResponse(status_code=400, content={"error": "question 필요"})

    async def event_generator():
        try:
            agent = create_agent()
            yield sse({"type": "start", "message": "에이전트 시작..."})

            async for update in agent.astream(
                {"messages": [HumanMessage(content=question)]},
                stream_mode="updates",
            ):
                # tool 호출 단계
                if "tools" in update:
                    for msg in update["tools"].get("messages", []):
                        content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content, ensure_ascii=False)
                        yield sse({"type": "tool_result", "tool": getattr(msg, "name", "unknown"), "content": content})

                # agent 사고 단계
                if "agent" in update:
                    for msg in update["agent"].get("messages", []):
                        if getattr(msg, "tool_calls", None):
                            for tc in msg.tool_calls:
                                yield sse({"type": "tool_call", "tool": tc["name"], "args": tc["args"]})
                        if isinstance(msg.content, str) and msg.content.strip():
                            yield sse({"type": "answer", "content": msg.content})

            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Agent 실행 (일반 JSON)
class AgentRunRequest(BaseModel):
    question: str


@app.post("/api/agent/run")
async def agent_run(body: AgentRunRequest):
    try:
        question = body.question
        if not question:
            return JSONResponse(status_code=400, content={"success": False, "error": "question 필요"})

        agent = create_agent()
        steps = []
        final_answer = ""

        async for update in agent.astream(
            {"messages": [HumanMessage(content=question)]},
            stream_mode="updates",
        ):
            if "tools" in update:
                for msg in update["tools"].get("messages", []):
                    content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content, ensure_ascii=False)
                    steps.append({"type": "tool_result", "tool": getattr(msg, "name", "tool"), "content": content})

            if "agent" in update:
                for msg in update["agent"].get("messages", []):
                    if getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            steps.append({"type": "tool_call", "tool": tc["name"], "args": tc["args"]})
                    if isinstance(msg.content, str) and msg.content.strip():
                        final_answer = msg.content
                        steps.append({"type": "answer", "content": msg.content})

        return {"success": True, "question": question, "answer": final_answer, "steps": steps}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


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
