from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = init_chat_model("gpt-4o-mini")

# 세 가지 관점에서 동시 분석
positive_prompt = ChatPromptTemplate.from_template(
    "{topic}의 긍정적인 측면 3가지를 설명하세요."
)

negative_prompt = ChatPromptTemplate.from_template(
    "{topic}의 부정적인 측면 3가지를 설명하세요."
)
neutral_prompt = ChatPromptTemplate.from_template(
    "{topic}에 대한 객관적인 현황을 설명하세요."
)
# 병렬 체인 구성
parallel_chain = RunnableParallel(
    positive=positive_prompt | llm | StrOutputParser(),
    negative=negative_prompt | llm | StrOutputParser(),
    neutral=neutral_prompt | llm | StrOutputParser(),
)

# 실행 (세 체인이 동시에 실행됨)
results = parallel_chain.invoke({"topic": "AI가 IT취업에 끼치는 영향"})

print("=== 긍정적 측면 ===")
print(results["positive"])
print("\n=== 부정적 측면 ===")
print(results["negative"])
print("\n=== 객관적 현황 ===")
print(results["neutral"])