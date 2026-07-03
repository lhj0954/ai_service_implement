from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = init_chat_model("gpt-4o-mini")

class Answer(BaseModel):
    topic: str = Field(description="주제")
    summary: str = Field(description="요약")

parser = PydanticOutputParser(pydantic_object=Answer)

# 프롬프트에 형식 지시문 자동 삽입
format_instructions = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_template(
    "{question}\n\n{format_instructions}"
).partial(format_instructions=format_instructions)

chain = prompt | model | parser
result = chain.invoke({"question": "LangChain을 한 줄로 설명해줘"})
print(result.topic, result.summary)