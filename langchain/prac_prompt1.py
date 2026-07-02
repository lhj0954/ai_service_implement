from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

load_dotenv()

llm = init_chat_model("gpt-4o-mini")

example_prompt = PromptTemplate.from_template("질문: {question}\n답변: {answer}")

examples = [
    {
        "question": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?",
        "answer": "지구 대기의 약 78%를 차지하는 질소입니다."
    },
    {
        "question": "광합성에 필요한 주요 요소들은 무엇인가요?",
        "answer": "광합성에 필요한 주요 요소는 빛, 이산화탄소, 물입니다."
    },
    {
        "question": "물의 화학식은 무엇인가요?",
        "answer": "물의 화학식은 H2O로, 수소 원자 2개와 산소 원자 1개로 이루어져 있습니다."
    },
    {
        "question": "인체에서 가장 큰 장기는 무엇인가요?",
        "answer": "인체에서 가장 큰 장기는 피부입니다."
    },
    {
        "question": "빛의 속도는 초당 몇 km인가요?",
        "answer": "빛의 속도는 진공 상태에서 초당 약 299,792km입니다."
    }
]

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="질문: {input}\n답변:",
    input_variables=["input"],
)

formatted_prompt = prompt.invoke({
    "input": "화성의 표면이 붉은 이유는 무엇인가요?"
})

response = llm.invoke(formatted_prompt)

print(response.content)