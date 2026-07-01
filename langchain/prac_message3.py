from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """도시 이름을 입력받아 해당 도시의 날씨 정보를 반환합니다."""
    weather_data = {
        "서울": "맑음, 25도",
        "부산": "흐림, 27도"
    }
    return weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다.")

@tool
def add_numbers(a: int, b: int) -> int:
    """두 정수를 더한 값을 반환합니다."""
    return a + b

tools = [get_weather, add_numbers]

llm = ChatOpenAI(model = "gpt-4o", temperature = 0)
llm_with_tools = llm.bind_tools(tools)

messages = [HumanMessage(content = "서울 날씨 알려줘")]

ai_msg = llm_with_tools.invoke(messages)
print("LLM이 호출하려는 tool 목록 : ", ai_msg.tool_calls)

messages.append(ai_msg)

tool_map = {"get_weather": get_weather, "add_numbers" : add_numbers}

for tool_call in ai_msg.tool_calls:
    selected_tool = tool_map[tool_call["name"]]
    tool_result = selected_tool.invoke(tool_call["args"])
    print(f"[Tool 실행 결과] {tool_call['name']} -> {tool_result}")

    messages.append(
        ToolMessage(content = str(tool_result), tool_call_id = tool_call["id"])
    )

final_response = llm_with_tools.invoke(messages)
print("최종 답변: ", final_response.content)