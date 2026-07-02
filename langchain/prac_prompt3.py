from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv()

llm = init_chat_model("gpt-4o-mini")

class CodeReview(BaseModel):
    """코드 리뷰 결과"""
    summary: str = Field(description="코드 기능 요약")
    score: int = Field(description="1-10점 품질 점수")
    issues: list[str] = Field(description="발견된 문제점 목록")
    suggestions: list[str] = Field(description="개선 제안 목록")

structured_llm = llm.with_structured_output(CodeReview)

result = structured_llm.invoke("""
다음 코드를 리뷰해주세요:

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 딕셔너리 형식의 메시지
messages = [
    {"role": "system", "content": "당신은 유용한 AI 어시스턴트입니다."},
    {"role": "user", "content": "파이썬이란?"},
    {"role": "assistant", "content": "파이썬은 프로그래밍 언어입니다."},
    {"role": "user", "content": "주요 특징은?"},
]

response = model.invoke(messages)
print(response.content)
""")

print(f"점수: {result.score}/10")
print(f"문제점: {result.issues}")