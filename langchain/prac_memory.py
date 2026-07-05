from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory


load_dotenv()

llm = init_chat_model(
    model="gpt-4o-mini",
    temperature = 0,
    max_tokens = 200
)

# 1. 세션별 메모리 저장소
store = {}
def get_session_history(session_id: str): # 세션별 히스토리를 관리하는 함수
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 AI 어시스턴트입니다."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chain = prompt | llm

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history , #세션 히스토리를 가져오는 함수
    input_messages_key="question", # 사용자 입력 키
    history_messages_key="history" # 대화 기록 키
)

#대화 실행
config = {"configurable": {"session_id": "user_123"}}

# 첫 번째 대화
response1 = chain_with_memory.invoke(
    {"question": "안녕하세요"},
    config=config
)
print("AI:", response1.content)
print("-" * 50)

# 두 번째 대화 (이전 대화 기억)
response2 = chain_with_memory.invoke(
    {"question": "내 이름은 이학준입니다."},
    config=config
)
print("AI:", response2.content)
print("-" * 50)

# 세 번째 대화 (이름 기억 확인)
response3 = chain_with_memory.invoke(
    {"question": "내 이름이 뭐였죠?"},
    config=config
)
print("AI:", response3.content)