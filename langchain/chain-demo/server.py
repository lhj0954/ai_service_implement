import os
import re
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
    RunnableBranch,
)
from langchain_openai import ChatOpenAI

load_dotenv()

app = FastAPI(title="LangChain Runnable 실습 (FastAPI)")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
parser = StrOutputParser()


# ──────────────────────────────────────────────
# Pydantic 요청 모델 (Express의 req.body 기본값과 동일)
# ──────────────────────────────────────────────
class SequenceRequest(BaseModel):
    topic: str = "양자컴퓨터"


class PassthroughRequest(BaseModel):
    question: str = "한국의 수도는?"


class LambdaRequest(BaseModel):
    text: str = "LangChain은 LLM 애플리케이션 개발 프레임워크입니다."


class ParallelRequest(BaseModel):
    topic: str = "인공지능"


class BranchRequest(BaseModel):
    text: str = "파이썬으로 피보나치 수열을 구현하는 방법을 알려주세요."


# ── 1. RunnableSequence ───────────────────────────
# 여러 Runnable을 순서대로 연결 (| 연산자 = RunnableSequence 명시적 체인 구성과 동일)
@app.post("/api/sequence")
async def sequence(body: SequenceRequest):
    try:
        topic = body.topic

        prompt = PromptTemplate.from_template(
            "{topic}에 대해 2문장으로 핵심만 설명해주세요."
        )

        # prompt | llm | parser 는 RunnableSequence.from([prompt, llm, parser])와 동일
        chain = prompt | llm | parser

        steps = [
            {"name": "PromptTemplate", "input": f'{{ topic: "{topic}" }}', "output": "PromptValue (포맷된 문자열)"},
            {"name": "ChatOpenAI", "input": "PromptValue", "output": "AIMessage"},
            {"name": "StringOutputParser", "input": "AIMessage", "output": "string"},
        ]

        result = await chain.ainvoke({"topic": topic})

        return {"success": True, "topic": topic, "result": result, "steps": steps}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ── 2. RunnablePassthrough ────────────────────────
# 입력을 그대로 다음 단계로 전달하거나, 기존 입력에 새 키를 추가(assign)
@app.post("/api/passthrough")
async def passthrough(body: PassthroughRequest):
    try:
        question = body.question

        prompt = PromptTemplate.from_template(
            "다음 질문에 한 문장으로 답해주세요.\n질문: {question}"
        )

        # RunnablePassthrough.assign()으로 입력 객체에 새 필드 추가
        chain = RunnablePassthrough.assign(
            question_length=RunnableLambda(lambda x: len(x["question"])),
            uppercased=RunnableLambda(lambda x: x["question"].upper()),
        ) | RunnablePassthrough.assign(
            answer=(
                RunnableLambda(lambda x: {"question": x["question"]})
                | prompt
                | llm
                | parser
            )
        )

        result = await chain.ainvoke({"question": question})

        return {
            "success": True,
            "question": question,
            "passthrough_result": {
                "question": result["question"],
                "question_length": result["question_length"],
                "uppercased": result["uppercased"],
                "answer": result["answer"],
            },
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ── 3. RunnableLambda ─────────────────────────────
# 일반 파이썬 함수를 Runnable로 래핑해 체인에 연결
@app.post("/api/lambda")
async def lambda_chain(body: LambdaRequest):
    try:
        text = body.text

        # 각 Lambda 단계 정의
        def _clean(input_: dict) -> dict:
            return {
                "original": input_["text"],
                "cleaned": re.sub(r"\s+", " ", input_["text"].strip()),
            }

        step1_clean = RunnableLambda(_clean)

        def _stats(input_: dict) -> dict:
            cleaned = input_["cleaned"]
            return {
                **input_,
                "char_count": len(cleaned),
                "word_count": len(cleaned.split(" ")),
            }

        step2_stats = RunnableLambda(_stats)

        step3_summarize = (
            RunnableLambda(lambda x: {"text": x["cleaned"]})
            | PromptTemplate.from_template('다음 문장을 10자 이내로 요약해주세요: "{text}"')
            | llm
            | parser
            | RunnableLambda(lambda summary: {"summary": summary})
        )

        async def _wrap(input_: dict) -> dict:
            out = await step3_summarize.ainvoke(input_)
            return {**input_, "summary": out["summary"]}

        step3_wrap = RunnableLambda(_wrap)

        chain = step1_clean | step2_stats | step3_wrap
        result = await chain.ainvoke({"text": text})

        return {"success": True, "result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ── 4. RunnableParallel ───────────────────────────
# 동일한 입력을 여러 체인에 동시에 병렬 실행
@app.post("/api/parallel")
async def parallel(body: ParallelRequest):
    try:
        topic = body.topic

        def make_chain(template: str):
            return PromptTemplate.from_template(template) | llm | parser

        parallel_chain = RunnableParallel(
            definition=make_chain("{topic}을(를) 한 문장으로 정의해주세요."),
            example=make_chain("{topic}의 실생활 적용 예시를 한 가지만 알려주세요."),
            future=make_chain("{topic}의 미래 전망을 한 문장으로 말해주세요."),
        )

        start = time.time()
        result = await parallel_chain.ainvoke({"topic": topic})
        elapsed = int((time.time() - start) * 1000)

        return {"success": True, "topic": topic, "result": result, "elapsed": elapsed}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ── 5. RunnableBranch ─────────────────────────────
# 조건에 따라 다른 체인을 실행 (if/else 분기)
@app.post("/api/branch")
async def branch_route(body: BranchRequest):
    try:
        text = body.text

        # 분류 체인: 입력 텍스트의 카테고리 판단
        classify_chain = (
            PromptTemplate.from_template(
                "다음 텍스트가 어떤 주제인지 한 단어로만 답하세요. "
                "(코딩/수학/과학/기타 중 하나)\n텍스트: {text}\n답:"
            )
            | llm
            | parser
            | RunnableLambda(lambda output: {"text": text, "category": output.strip()})
        )

        # 분기별 응답 체인
        coding_chain = (
            PromptTemplate.from_template("개발자 관점에서 답해주세요 (코드 예시 포함, 3문장): {text}")
            | llm
            | parser
        )
        math_chain = (
            PromptTemplate.from_template("수학적으로 단계별로 설명해주세요 (3문장): {text}")
            | llm
            | parser
        )
        science_chain = (
            PromptTemplate.from_template("과학적 원리를 중심으로 설명해주세요 (3문장): {text}")
            | llm
            | parser
        )
        default_chain = (
            PromptTemplate.from_template("친절하게 2문장으로 답해주세요: {text}")
            | llm
            | parser
        )

        # 분류 → 분기 처리
        classify_result = await classify_chain.ainvoke({"text": text})
        category = classify_result["category"]

        branch = RunnableBranch(
            (lambda x: bool(re.search(r"코딩|코드|프로그래밍", x["category"], re.I)), coding_chain),
            (lambda x: bool(re.search(r"수학", x["category"], re.I)), math_chain),
            (lambda x: bool(re.search(r"과학", x["category"], re.I)), science_chain),
            default_chain,
        )

        branch_input = {"text": text, "category": category}
        answer = await branch.ainvoke(branch_input)

        if re.search(r"코딩|코드|프로그래밍", category, re.I):
            matched_branch = "codingChain"
        elif re.search(r"수학", category, re.I):
            matched_branch = "mathChain"
        elif re.search(r"과학", category, re.I):
            matched_branch = "scienceChain"
        else:
            matched_branch = "defaultChain"

        return {
            "success": True,
            "text": text,
            "category": category,
            "matched_branch": matched_branch,
            "answer": answer,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# express.static("public")과 동일 — 반드시 API 라우트 아래에 마운트
app.mount("/", StaticFiles(directory="public", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 3000))
    print(f"✅ http://localhost:{port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
