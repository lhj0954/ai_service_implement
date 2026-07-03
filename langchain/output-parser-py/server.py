import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    CommaSeparatedListOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_openai import ChatOpenAI

load_dotenv()

app = FastAPI(title="LangChain Output Parser 실습 (Python)")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


# ─────────────────────────────────────────────
# 요청 바디 스키마
# ─────────────────────────────────────────────
class StringReq(BaseModel):
    input: str = "판교는 어떤 도시인가요?"


class JsonReq(BaseModel):
    topic: str = "서울"


class StructuredReq(BaseModel):
    movie: str = "인터스텔라"


class CsvReq(BaseModel):
    category: str = "한국 전통 음식"


def err_response(e: Exception, status_code: int = 500) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"success": False, "error": str(e)})


# ─────────────────────────────────────────────
# 1. StrOutputParser  (JS: StringOutputParser)
# ─────────────────────────────────────────────
@app.post("/api/string")
async def api_string(body: StringReq):
    try:
        input_ = body.input

        parser = StrOutputParser()
        prompt = PromptTemplate.from_template(
            "다음 질문에 2문장으로 간결하게 답해주세요.\n질문: {input}"
        )

        # raw AIMessage
        raw_chain = prompt | llm
        raw = await raw_chain.ainvoke({"input": input_})

        # with parser
        chain = prompt | llm | parser
        result = await chain.ainvoke({"input": input_})

        raw_dump = raw.model_dump()
        raw_keys = [
            k for k in ["id", "content", "tool_calls", "usage_metadata"] if k in raw_dump
        ]

        return {
            "success": True,
            "input": input_,
            "raw_type": raw.__class__.__name__,
            "raw_keys": raw_keys,
            "parsed_type": type(result).__name__,
            "parsed_result": result,
        }
    except Exception as e:
        return err_response(e)


# ─────────────────────────────────────────────
# 2. JsonOutputParser
# ─────────────────────────────────────────────
@app.post("/api/json")
async def api_json(body: JsonReq):
    try:
        topic = body.topic

        parser = JsonOutputParser()
        prompt = PromptTemplate.from_template(
            """다음 주제에 대한 정보를 JSON 형식으로 반환하세요.
마크다운 코드블록 없이 순수 JSON만 출력하세요.
형식: {{"name":"이름","description":"한 줄 설명","population":숫자,"features":["특징1","특징2","특징3"]}}
주제: {topic}"""
        )

        chain = prompt | llm | parser
        result = await chain.ainvoke({"topic": topic})

        return {
            "success": True,
            "topic": topic,
            "parsed_type": type(result).__name__,
            "parsed_result": result,
        }
    except Exception as e:
        return err_response(e)


# ─────────────────────────────────────────────
# 3. PydanticOutputParser  (JS: StructuredOutputParser.fromZodSchema)
#    → Zod 스키마와 동일하게 "타입까지 검증"하는 파서이므로 Pydantic으로 대체
# ─────────────────────────────────────────────
class MovieInfo(BaseModel):
    title: str = Field(description="영화 제목")
    director: str = Field(description="감독 이름")
    year: int = Field(description="개봉 연도")
    genre: list[str] = Field(description="장르 목록")
    rating: float = Field(ge=0, le=10, description="평점 (0~10)")
    summary: str = Field(description="줄거리 한 줄 요약")


@app.post("/api/structured")
async def api_structured(body: StructuredReq):
    try:
        movie = body.movie

        parser = PydanticOutputParser(pydantic_object=MovieInfo)
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate.from_template(
            "다음 영화에 대한 정보를 제공해주세요.\n{format_instructions}\n영화: {movie}"
        )

        chain = prompt | llm | parser
        result: MovieInfo = await chain.ainvoke(
            {"movie": movie, "format_instructions": format_instructions}
        )

        return {
            "success": True,
            "movie": movie,
            "format_instructions": format_instructions,
            "parsed_result": result.model_dump(),
        }
    except Exception as e:
        return err_response(e)


# ─────────────────────────────────────────────
# 4. CommaSeparatedListOutputParser
# ─────────────────────────────────────────────
@app.post("/api/csvlist")
async def api_csvlist(body: CsvReq):
    try:
        category = body.category

        parser = CommaSeparatedListOutputParser()
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate.from_template(
            "다음 카테고리의 항목을 5개 나열해주세요.\n{format_instructions}\n카테고리: {category}"
        )

        chain = prompt | llm | parser
        result: list[str] = await chain.ainvoke(
            {"category": category, "format_instructions": format_instructions}
        )

        return {
            "success": True,
            "category": category,
            "format_instructions": format_instructions,
            "parsed_type": "list[str]",
            "parsed_result": result,
            "count": len(result),
        }
    except Exception as e:
        return err_response(e)


# ─────────────────────────────────────────────
# 정적 파일 서빙 (express.static("public") 대응)
# ※ API 라우트를 먼저 등록한 뒤 마지막에 마운트
# ─────────────────────────────────────────────
app.mount("/", StaticFiles(directory="public", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 3000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
