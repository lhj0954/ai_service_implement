import json
import time
from datetime import datetime
from pathlib import Path
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()  # .env 파일에서 OPENAI_API_KEY 등을 읽어옴

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
# index.html 등 정적 파일 제공 (server.js의 app.use(express.static("public"))와 동일)
app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# index.html은 /에서 직접 반환
@app.get("/")
def read_index():
    return FileResponse(PUBLIC_DIR / "index.html")

# ── 요청 바디 스키마 (Pydantic) ───────────────────────
class FewShotRequest(BaseModel):
    input: str


class PartialRequest(BaseModel):
    phenomenon: str


# ── 1. FewShotPromptTemplate ──────────────────────────
@app.post("/api/fewshot")
async def fewshot(req: FewShotRequest):
    try:
        example_prompt = PromptTemplate.from_template("질문: {question}\n{answer}")

        examples = [
            {"question": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?",
             "answer": "지구 대기의 약 78%를 차지하는 질소입니다."},
            {"question": "광합성에 필요한 주요 요소들은 무엇인가요?",
             "answer": "광합성에 필요한 주요 요소는 빛, 이산화탄소, 물입니다."},
            {"question": "피타고라스 정리를 설명해주세요.",
             "answer": "피타고라스 정리는 직각삼각형에서 빗변의 제곱이 다른 두 변의 제곱의 합과 같다는 것입니다."},
            {"question": "지구의 자전 주기는 얼마인가요?",
             "answer": "지구의 자전 주기는 약 24시간(정확히는 23시간 56분 4초)입니다."},
            {"question": "DNA의 기본 구조를 간단히 설명해주세요.",
             "answer": "DNA는 두 개의 폴리뉴클레오티드 사슬이 이중 나선 구조를 이루고 있습니다."},
            {"question": "원주율(π)의 정의는 무엇인가요?",
             "answer": "원주율(π)은 원의 지름에 대한 원의 둘레의 비율입니다."},
        ]

        # LLM에게 보여줄 예제들 + 실제 질문(suffix)을 합쳐 최종 prompt 구성
        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            suffix="질문: {input}",
            input_variables=["input"],
        )

        formatted_prompt = prompt.invoke({"input": req.input})
        prompt_text = formatted_prompt.to_string()

        # 사용자 입력 → FewShotPromptTemplate → 완성된 Prompt → LLM → AIMessage
        chain = prompt | llm
        response = chain.invoke({"input": req.input})

        return {"success": True, "prompt": prompt_text, "answer": response.content}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── 2. Partial Prompt (prompt.partial) ────────────────
def get_current_season() -> str:
    month = datetime.now().month
    if 3 <= month <= 5:
        return "봄"
    elif 6 <= month <= 8:
        return "여름"
    elif 9 <= month <= 11:
        return "가을"
    else:
        return "겨울"


@app.post("/api/partial")
async def partial(req: PartialRequest):
    try:
        season = get_current_season()

        prompt = PromptTemplate.from_template(
            "{season}에 일어나는 대표적인 지구과학 현상은 {phenomenon}입니다."
        )
        # season 변수를 미리 고정 → 남은 변수는 phenomenon 하나뿐
        partial_prompt = prompt.partial(season=season)
        result = partial_prompt.invoke({"phenomenon": req.phenomenon})
        prompt_text = result.to_string()

        # LLM에도 전달해서 관련 설명 생성
        explain_prompt = PromptTemplate.from_template(
            "{season}에 일어나는 대표적인 지구과학 현상인 {phenomenon}에 대해 2~3문장으로 설명해주세요."
        )
        explain_partial = explain_prompt.partial(season=season)
        chain = explain_partial | llm
        response = chain.invoke({"phenomenon": req.phenomenon})

        return {
            "success": True,
            "season": season,
            "prompt": prompt_text,
            "answer": response.content,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── 3. Stream (SSE) ────────────────────────────────────
@app.get("/api/stream")
async def stream(request: Request):
    user_input = request.query_params.get("input", "블랙홀이란 무엇인가요?")

    async def event_generator():
        try:
            prompt = PromptTemplate.from_template("다음 질문에 3문장으로 답해주세요: {input}")
            chain = prompt | llm

            async for chunk in chain.astream({"input": user_input}):
                text = chunk.content or ""
                yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ── 4. Batch ────────────────────────────────────────────
@app.post("/api/batch")
async def batch():
    try:
        questions = [
            "태양계에서 가장 큰 행성은?",
            "물의 화학식은?",
            "빛의 속도는 얼마인가요?",
        ]

        prompt = PromptTemplate.from_template("다음 질문에 한 문장으로 간단히 답해주세요: {input}")
        chain = prompt | llm

        start_time = time.time()
        responses = await chain.abatch([{"input": q} for q in questions])
        elapsed = int((time.time() - start_time) * 1000)  # ms 단위

        return {
            "success": True,
            "results": [
                {"question": q, "answer": r.content}
                for q, r in zip(questions, responses)
            ],
            "elapsed": elapsed,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
