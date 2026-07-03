from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv()

# 출력 스키마 정의
class MovieReview(BaseModel):
    """영화 리뷰 정보"""
    title: str = Field(description="영화 제목")
    rating: int = Field(description="1-10점 평점")
    pros: list[str] = Field(description="장점 목록")
    cons: list[str] = Field(description="단점 목록")
    summary: str = Field(description="한 줄 요약")

llm = init_chat_model("gpt-4o-mini")
structured_llm = llm.with_structured_output(MovieReview, include_raw=True)

# 실행
review = structured_llm.invoke("인셉션 영화를 리뷰해주세요.")
review = review["parsed"]
print(f"제목: {review.title}")
print(f"평점: {review.rating}/10")
print(f"장점: {review.pros}")
print(f"단점: {review.cons}")
print(f"요약: {review.summary}")