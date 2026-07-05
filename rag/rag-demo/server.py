"""
1. InMemory RAG          — 내장 문서로 즉시 실습
2. Web RAG                — URL을 로드해서 임베딩
3. CSV RAG                 — CSV 파일 업로드
4. PDF RAG                 — PDF 파일 업로드
5. Conversational RAG   — 대화 이력을 반영한 검색 쿼리 재작성 + RAG
"""

import asyncio
import math
import os
import uuid
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("USER_AGENT", "langchain-rag-demo/1.0")

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

app = FastAPI(title="LangChain RAG 실습 5종 (FastAPI)")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════
#  SimpleVectorStore — 순수 파이썬 코사인 유사도
# ════════════════════════════════════════════════
class SimpleVectorStore:
    def __init__(self):
        self.docs: List[dict] = []  # [{page_content, metadata, embedding}]

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb + 1e-10)

    async def add_documents(self, docs: List[Document], embedder: OpenAIEmbeddings) -> int:
        texts = [d.page_content for d in docs]
        embeddings = await embedder.aembed_documents(texts)
        for doc, emb in zip(docs, embeddings):
            self.docs.append({"page_content": doc.page_content, "metadata": doc.metadata, "embedding": emb})
        return len(self.docs)

    async def similarity_search(self, query: str, k: int, embedder: OpenAIEmbeddings) -> List[Document]:
        q_emb = await embedder.aembed_query(query)
        scored = [(self._cosine_sim(q_emb, d["embedding"]), d) for d in self.docs]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [Document(page_content=d["page_content"], metadata=d["metadata"]) for _, d in scored[:k]]

    def clear(self) -> None:
        self.docs = []


# ── 공통 RAG 체인 빌더 ────────────────────────────
def build_rag_chain(vector_store: SimpleVectorStore, embedder: OpenAIEmbeddings, k: int = 3):
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", '다음 컨텍스트만 사용해서 한국어로 답변해.\n모르면 "모르겠어요"라고 말해.\n\n컨텍스트:\n{context}'),
            ("human", "{input}"),
        ]
    )
    answer_chain = prompt | model | StrOutputParser()

    # 원본 JS는 context/source_docs를 각각 별도 similaritySearch()로 구했지만,
    # (동일한 검색을 두 번 하는 중복 호출이라) 여기서는 한 번의 검색 결과를 재사용하도록 개선했습니다.
    async def retrieve(inp: dict) -> dict:
        docs = await vector_store.similarity_search(inp["input"], k, embedder)
        return {
            **inp,
            "context": "\n\n".join(d.page_content for d in docs),
            "source_docs": [
                {"content": d.page_content[:200], "source": d.metadata.get("source", "-")} for d in docs
            ],
        }

    async def generate(inp: dict) -> dict:
        answer = await answer_chain.ainvoke({"input": inp["input"], "context": inp["context"]})
        return {"answer": answer, "source_docs": inp["source_docs"]}

    return RunnableLambda(retrieve) | RunnableLambda(generate)


def make_embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


class QueryRequest(BaseModel):
    input: Optional[str] = None


# ════════════════════════════════════════════════
#  1. InMemory RAG
# ════════════════════════════════════════════════
mem_store = SimpleVectorStore()
mem_ready = False

MEM_DOCS = [
    "LangChain은 LLM 기반 애플리케이션을 쉽게 구축할 수 있도록 도와주는 오픈소스 프레임워크입니다.",
    "LCEL(LangChain Expression Language)은 체인을 선언적으로 구성하는 방법으로, pipe() 연산자로 컴포넌트를 연결합니다.",
    "RAG(Retrieval-Augmented Generation)는 외부 문서를 검색해 LLM의 답변 품질을 높이는 기법입니다.",
    "벡터스토어는 텍스트를 임베딩 벡터로 변환해 저장하고 의미적 유사도 기반 검색을 가능하게 합니다.",
    "RunnableSequence는 여러 Runnable을 순서대로 연결하는 클래스입니다.",
    "RunnablePassthrough는 입력을 그대로 통과시키거나 assign()으로 새 키를 추가하는 클래스입니다.",
    "ChatMessageHistory는 대화 이력을 세션별로 저장하고 관리하는 클래스입니다.",
    "MessagesPlaceholder는 프롬프트 템플릿에서 대화 이력을 동적으로 삽입할 때 사용합니다.",
    "OpenAIEmbeddings는 텍스트를 OpenAI의 임베딩 모델로 벡터화하는 클래스입니다.",
    "RecursiveCharacterTextSplitter는 문서를 청크 크기와 오버랩 기준으로 분할하는 클래스입니다.",
]


@app.post("/api/mem/init")
async def mem_init():
    global mem_ready
    try:
        mem_store.clear()
        docs = [Document(page_content=t, metadata={"source": f"mem-{i + 1}"}) for i, t in enumerate(MEM_DOCS)]
        count = await mem_store.add_documents(docs, make_embedder())
        mem_ready = True
        return {"success": True, "doc_count": count}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/mem/query")
async def mem_query(body: QueryRequest):
    if not body.input:
        return JSONResponse(status_code=400, content={"success": False, "error": "input 필요"})
    if not mem_ready:
        return JSONResponse(status_code=400, content={"success": False, "error": "먼저 초기화하세요."})
    try:
        result = await build_rag_chain(mem_store, make_embedder()).ainvoke({"input": body.input})
        return {"success": True, **result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ════════════════════════════════════════════════
#  2. Web RAG (Cheerio → WebBaseLoader/BeautifulSoup)
# ════════════════════════════════════════════════
web_store = SimpleVectorStore()
web_ready = False


class WebLoadRequest(BaseModel):
    url: str = "https://ko.wikipedia.org/wiki/%EC%9C%A4%EB%8F%99%EC%A3%BC"


@app.post("/api/web/load")
async def web_load(body: WebLoadRequest):
    global web_ready
    try:
        loader = WebBaseLoader(body.url)
        # WebBaseLoader.load()는 동기(blocking) I/O이므로 스레드풀에서 실행해 이벤트 루프를 막지 않음
        docs = await asyncio.to_thread(loader.load)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        web_store.clear()
        chunk_count = await web_store.add_documents(chunks, make_embedder())
        web_ready = True
        return {"success": True, "url": body.url, "chunk_count": chunk_count}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/web/query")
async def web_query(body: QueryRequest):
    if not body.input:
        return JSONResponse(status_code=400, content={"success": False, "error": "input 필요"})
    if not web_ready:
        return JSONResponse(status_code=400, content={"success": False, "error": "먼저 URL을 로드하세요."})
    try:
        result = await build_rag_chain(web_store, make_embedder()).ainvoke({"input": body.input})
        return {"success": True, **result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ════════════════════════════════════════════════
#  3. CSV RAG
# ════════════════════════════════════════════════
csv_store = SimpleVectorStore()
csv_ready = False


@app.post("/api/csv/load")
async def csv_load(file: Optional[UploadFile] = File(default=None)):
    global csv_ready
    if file is None:
        return JSONResponse(status_code=400, content={"success": False, "error": "CSV 파일을 업로드하세요."})

    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    try:
        tmp_path.write_bytes(await file.read())

        loader = CSVLoader(str(tmp_path))
        docs = await asyncio.to_thread(loader.load)
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        csv_store.clear()
        count = await csv_store.add_documents(chunks, make_embedder())
        csv_ready = True
        return {"success": True, "filename": file.filename, "chunk_count": count}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/csv/query")
async def csv_query(body: QueryRequest):
    if not body.input:
        return JSONResponse(status_code=400, content={"success": False, "error": "input 필요"})
    if not csv_ready:
        return JSONResponse(status_code=400, content={"success": False, "error": "먼저 CSV를 업로드하세요."})
    try:
        result = await build_rag_chain(csv_store, make_embedder()).ainvoke({"input": body.input})
        return {"success": True, **result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ════════════════════════════════════════════════
#  4. PDF RAG
# ════════════════════════════════════════════════
pdf_store = SimpleVectorStore()
pdf_ready = False


@app.post("/api/pdf/load")
async def pdf_load(file: Optional[UploadFile] = File(default=None)):
    global pdf_ready
    if file is None:
        return JSONResponse(status_code=400, content={"success": False, "error": "PDF 파일을 업로드하세요."})

    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    try:
        tmp_path.write_bytes(await file.read())

        loader = PyPDFLoader(str(tmp_path))
        docs = await asyncio.to_thread(loader.load)  # docs 1개 = PDF 1페이지 (원본 JS PDFLoader와 동일)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        pdf_store.clear()
        count = await pdf_store.add_documents(chunks, make_embedder())
        pdf_ready = True
        return {
            "success": True,
            "filename": file.filename,
            "chunk_count": count,
            "page_count": len(docs),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/pdf/query")
async def pdf_query(body: QueryRequest):
    if not body.input:
        return JSONResponse(status_code=400, content={"success": False, "error": "input 필요"})
    if not pdf_ready:
        return JSONResponse(status_code=400, content={"success": False, "error": "먼저 PDF를 업로드하세요."})
    try:
        result = await build_rag_chain(pdf_store, make_embedder(), k=5).ainvoke({"input": body.input})
        return {"success": True, **result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ════════════════════════════════════════════════
#  5. Conversational RAG
#  chat history + retriever + RunnableWithMessageHistory
# ════════════════════════════════════════════════
conv_rag_store = SimpleVectorStore()
conv_rag_ready = False
conv_sessions: dict[str, InMemoryChatMessageHistory] = {}


def get_conv_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in conv_sessions:
        conv_sessions[session_id] = InMemoryChatMessageHistory()
    return conv_sessions[session_id]


CONV_DOCS = [
    "윤동주(1917~1945)는 일제강점기 시인으로, 저항시인으로 불린다.",
    "윤동주의 대표작으로는 '서시', '자화상', '별 헤는 밤', '쉽게 씌어진 시' 등이 있다.",
    "윤동주의 시집 '하늘과 바람과 별과 시'는 사후에 출판되었다.",
    "윤동주는 연희전문학교를 졸업하고 일본 도시샤 대학에 유학했다.",
    "윤동주는 1945년 2월 후쿠오카 형무소에서 순국하였다.",
    "LangChain의 ConversationalRAG는 대화 이력을 기억하며 문서 기반 답변을 제공한다.",
    "RunnableWithMessageHistory는 대화 이력을 자동으로 관리하는 LangChain 클래스다.",
    "MessagesPlaceholder는 프롬프트에 대화 이력을 동적으로 삽입하는 클래스다.",
    "RAG 체인에서 retriever는 사용자 질문과 유사한 문서를 벡터 검색으로 찾는다.",
    "Conversational RAG는 현재 질문과 대화 이력을 함께 고려해 검색 쿼리를 생성한다.",
]


@app.post("/api/convrag/init")
async def convrag_init():
    global conv_rag_ready
    try:
        conv_rag_store.clear()
        docs = [Document(page_content=t, metadata={"source": f"conv-{i + 1}"}) for i, t in enumerate(CONV_DOCS)]
        count = await conv_rag_store.add_documents(docs, make_embedder())
        conv_rag_ready = True
        return {"success": True, "doc_count": count}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


class ConvChatRequest(BaseModel):
    input: Optional[str] = None
    sessionId: str = "default"


@app.post("/api/convrag/chat")
async def convrag_chat(body: ConvChatRequest):
    if not body.input:
        return JSONResponse(status_code=400, content={"success": False, "error": "input 필요"})
    if not conv_rag_ready:
        return JSONResponse(status_code=400, content={"success": False, "error": "먼저 초기화하세요."})

    try:
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        # 이전 대화 이력 가져오기
        history = get_conv_history(body.sessionId).messages
        hist_text = "\n".join(f"{'사용자' if m.type == 'human' else 'AI'}: {m.content}" for m in history)

        # 대화 이력을 고려한 검색 쿼리 생성
        # 주의: 대화 내용을 f-string으로 프롬프트에 직접 끼워넣으면 사용자가 '{'나 '}' 를 입력했을 때
        # ChatPromptTemplate이 이를 템플릿 변수로 오인해 깨질 수 있어, 여기서는 메시지 객체를 직접 구성해서 피했습니다.
        search_query = body.input
        if history:
            from langchain_core.messages import HumanMessage, SystemMessage

            query_messages = [
                SystemMessage(content="대화 이력을 참고해서 현재 질문의 핵심 검색어를 한 문장으로 만들어줘. 검색어만 출력해."),
                HumanMessage(content=f"대화 이력:\n{hist_text}\n\n현재 질문: {body.input}"),
            ]
            query_response = await (model | StrOutputParser()).ainvoke(query_messages)
            search_query = query_response

        # 검색
        retrieved_docs = await conv_rag_store.similarity_search(search_query, 3, make_embedder())
        context = "\n\n".join(d.page_content for d in retrieved_docs)
        source_docs = [
            {"content": d.page_content[:150], "source": d.metadata.get("source", "-")} for d in retrieved_docs
        ]

        # Conversational RAG 프롬프트
        conv_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    '당신은 친절한 AI 어시스턴트입니다. 이전 대화를 기억하며 아래 컨텍스트를 참고해 답변하세요.\n'
                    '모르면 "모르겠어요"라고 말하세요.\n\n컨텍스트:\n{context}',
                ),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )

        base_chain = conv_prompt | model | StrOutputParser()

        conv_chain = RunnableWithMessageHistory(
            base_chain,
            get_session_history=get_conv_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        answer = await conv_chain.ainvoke(
            {"input": body.input, "context": context},
            config={"configurable": {"session_id": body.sessionId}},
        )

        msgs = get_conv_history(body.sessionId).messages
        chat_history = [{"role": "human" if m.type == "human" else "ai", "content": m.content} for m in msgs]

        return {
            "success": True,
            "input": body.input,
            "answer": answer,
            "search_query": search_query,
            "source_docs": source_docs,
            "history": chat_history,
            "sessionId": body.sessionId,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


class ConvResetRequest(BaseModel):
    sessionId: str = "default"


@app.post("/api/convrag/reset")
async def convrag_reset(body: ConvResetRequest):
    try:
        if body.sessionId in conv_sessions:
            conv_sessions[body.sessionId].clear()
        return {"success": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# express.static("public")과 동일 — 반드시 API 라우트 아래에 마운트
app.mount("/", StaticFiles(directory="public", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 3000))
    print(f"✅ http://localhost:{port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
