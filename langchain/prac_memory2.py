from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

# 체크포인터로 상태 저장
checkpointer = InMemorySaver()

agent_with_memory = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    checkpointer=checkpointer  # 자동 상태 저장
)

# thread_id로 대화 구분
config = {"configurable": {"thread_id": "conversation_1"}}
# 첫 번째 대화
response1 = agent_with_memory.invoke(
    {"messages":[{"role": "user", "content": "안녕하세요"}]}, 
    config=config
)
print("AI:", response1['messages'][-1].content)
print("-" * 50)

# 두 번째 대화 (이전 대화 기억)
response2 = agent_with_memory.invoke(
    {"messages":[{"role": "user", "content": "내 이름은 이학준입니다."}]}, 
    config=config
)
print("AI:", response2['messages'][-1].content)
print("-" * 50)

# 세 번째 대화 (이름 기억 확인)
response3 = agent_with_memory.invoke(
    {"messages":[{"role": "user", "content": "내 이름이 뭐였죠?"}]}, 
    config=config
)
print("AI:", response3['messages'][-1].content)