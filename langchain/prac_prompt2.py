from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = init_chat_model("gpt-4o-mini")

# 표 형식 요청
table_prompt = ChatPromptTemplate.from_messages([
("system", "응답은 마크다운 표 형식으로 작성하세요."),
("human", """다음 프로그래밍 언어들을 비교해주세요: {languages}
| 언어 | 주요 용도 | 장점 | 단점 | 학습 난이도 |
형식으로 작성해주세요.""")
])

# JSON 형식 요청
json_prompt = ChatPromptTemplate.from_messages([
    ("system", "응답은 유효한 JSON 형식으로만 작성하세요. 다른 텍스트는 포함하지 마세요."),
    ("human", """다음 텍스트에서 정보를 추출하세요: {text}
형식:
{{
    "name": "이름",
    "email": "이메일",
    "phone": "전화번호"
}}""")
])

# 번호 목록 형식 요청
list_prompt = ChatPromptTemplate.from_messages([
    ("system", "응답은 번호 목록으로 작성하세요. 각 항목은 한 줄로 간결하게."),
    ("human", "{topic}의 장점 5가지를 알려주세요.")
])

table_chain =  table_prompt | llm
json_chain =  json_prompt | llm
list_chain =  list_prompt | llm

print("="*50)
print("1. 표 형식 결과")
print("="*50)
table_response = table_chain.invoke({"languages": "Python, JavaScript, Java"})
print(table_response.content)

print("="*50)
print("2. JSON 형식 결과")
print("="*50)
sample_text = "안녕하세요, 저는 이학준입니다. 이메일은 hjlee0954@gmail.com이고 전화번호는 010-1234-1234입니다"
json_response = json_chain.invoke({"text" : sample_text})
print(json_response.content)

print("="*50)
print("3. 번호 목록 형식 결과")
print("="*50)
sample_topic = "맥북"
list_response = list_chain.invoke({"topic" : sample_topic})
print(list_response.content)