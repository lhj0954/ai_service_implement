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