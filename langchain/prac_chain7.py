from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv()

llm = init_chat_model("gpt-4o-mini")

# Pydantic 모델 정의
class BookInfo(BaseModel):
    title: str = Field(description="책 제목")
    author: str = Field(description="저자 이름")
    year: int = Field(description="출판 연도")
    genre: str = Field(description="장르")

# JSON 파서 생성
parser = JsonOutputParser(pydantic_object=BookInfo)
# 포맷 지시사항 확인

print(parser.get_format_instructions())

# 프롬프트 구성
prompt = PromptTemplate(
    template="다음 책에 대한 정보를 제공해주세요: {book_name}\n{format_instructions}",
    input_variables=["book_name"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 체인 구성
llm = init_chat_model("gpt-4o-mini", temperature=0)
chain = prompt | llm | parser

# 실행
result = chain.invoke({"book_name": "삼체"})
print(result)