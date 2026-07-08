from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv

load_dotenv()

# 1. 모델 초기화
model = init_chat_model("openai:gpt-4o-mini", temperature=0)

# 2. langchain에 내장된 도구tool 정의
tavily = TavilySearch(max_results=3)

class State(TypedDict):
    role: str
    content : str

@tool
def calculator(expression: str) -> dict:
    """수학 계산을 수행합니다. 사칙연산, 거듭제곱, 괄호를 지원합니다.
    Args:
        expression: 계산할 수학 표현식
    """
    import re
    if not re.match(r'^[0-9+\-*/().%\s]+$', state['content']):
        return "오류: 허용되지 않는 문자가 포함되어 있습니다."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {str(e)}"

@tool
def get_current_datetime() -> dict:
    """현재 날짜와 시간을 반환합니다."""
    from datetime import datetime
    return datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분 %S초 (%A)")

# 3. 에이전트 생성
agent = create_agent(
    model=model,
    tools=[tavily, calculator, get_current_datetime],
    system_prompt="당신은 도움이 되는 AI 어시스턴트입니다.",
    checkpointer=InMemorySaver(),
)

builder = StateGraph(State)
builder.add_node("calculator", calculator)
builder.add_node("get_current_datetime", get_current_datetime)

builder.add_edge(START, "calculator")
builder.add_edge("calculator", "get_current_datetime")
builder.add_edge("get_current_datetime", END)

# 4. 실행
config = {"configurable": {"thread_id": "session-1"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "판교 최고의 식당은 어디인가?"}]},
    config=config,
)

print(result["messages"][-1].content)