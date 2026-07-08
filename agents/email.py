"""
Human-in-the-loop ? 위험한 작업 실행 전 사람의 승인을 받는 Agent  
====================================================================================
langchain.agents.create_agent() + HumanInTheLoopMiddleware로 다시 작성했습니다.

핵심 개념
- create_agent(model, tools, middleware=[...]) 로 Agent를 만들면,
  Middleware가 알아서 "모델 호출 → Tool 실행" 루프 사이에 끼어들어 동작합니다.
- HumanInTheLoopMiddleware(interrupt_on={...}) 를 넘기면, 지정한 Tool이 호출되려는 순간
  자동으로 그래프 실행이 멈춥니다 (내부적으로 LangGraph의 interrupt()를 사용).
  → 이전 버전에서 직접 짰던 "tool_node 직전에 멈추는" 로직을 middleware 한 줄로 대체한 것입니다.
- 멈춘 지점은 checkpointer(MemorySaver)가 기억하고 있으므로,
  Command(resume={"decisions": [...]}) 로 재개하면 승인/거부 결과에 따라 이어서 실행됩니다.

 
"""

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()


# ------------------------------------------------------------
# 1. "위험한" Tool ? 실제로는 이메일을 보내는 등 되돌리기 어려운 작업
# ------------------------------------------------------------
@tool
def send_email(to: str, subject: str) -> str:
    """이메일을 실제로 발송합니다. (되돌릴 수 없는 위험한 작업이므로 사람의 승인이 필요합니다)

    Args:
        to: 수신자 이메일 주소
        subject: 이메일 제목
    """
    return f"? '{to}' 에게 '{subject}' 제목으로 이메일을 발송했습니다."


# ------------------------------------------------------------
# 2. Agent 구성 ? create_agent + HumanInTheLoopMiddleware
# ------------------------------------------------------------
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
checkpointer = MemorySaver()

agent = create_agent(
    model=model,
    tools=[send_email],
    system_prompt="당신은 이메일 업무를 도와주는 AI 비서입니다. 한국어로 답변하세요.",
    checkpointer=checkpointer,
    middleware=[
        HumanInTheLoopMiddleware(
            # send_email Tool이 호출되려고 할 때마다 사람에게 승인/거부를 물어봄
            interrupt_on={
                "send_email": {
                    "allowed_decisions": ["approve", "reject"],
                }
            },
            description_prefix="다음 작업을 실행하기 전 승인이 필요합니다",
        )
    ],
)


# ------------------------------------------------------------
# 3. 사람에게 승인 여부를 물어보는 헬퍼
# ------------------------------------------------------------
def ask_human_decisions(interrupt_payload: dict) -> dict:
    """HumanInTheLoopMiddleware가 만든 HITLRequest(action_requests)를 보여주고,
    터미널 입력으로 승인/거부를 받아 resume용 decisions를 만들어 반환합니다."""
    decisions = []
    for action in interrupt_payload["action_requests"]:
        print("\n?  실행 정지됨 ? 사람의 승인이 필요한 작업이 있습니다.")
        print(f"   Tool : {action['name']}")
        print(f"   Args : {action['args']}")
        print(f"   설명 : {action.get('description', '')}")

        answer = input("   승인하시겠습니까? (y=승인 / n=거부): ").strip().lower()
        if answer in ("y", "yes", "승인"):
            decisions.append({"type": "approve"})
        else:
            reason = input("   거부 사유(엔터 시 기본 메시지 사용): ").strip()
            decision = {"type": "reject"}
            if reason:
                decision["message"] = reason
            decisions.append(decision)

    return {"decisions": decisions}


# ------------------------------------------------------------
# 4. 실행 ? 사용자로부터 입력을 받아 계속 대화
# ------------------------------------------------------------
def run_interactive():
    thread_id = input("대화 세션 ID를 입력하세요 (그냥 엔터 시 'default'): ").strip() or "default"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n[thread_id={thread_id}] 대화를 시작합니다. 종료하려면 'exit' 또는 'quit'을 입력하세요.\n")

    while True:
        user_input = input("나: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("대화를 종료합니다.")
            break
        if not user_input:
            continue

        # 1) 사용자의 메시지로 Agent 실행
        result = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

        # 2) 승인이 필요한 Tool 호출이 있어서 멈췄는지 확인 (반복적으로 발생 가능)
        while "__interrupt__" in result:
            interrupt_payload = result["__interrupt__"][0].value
            resume_payload = ask_human_decisions(interrupt_payload)
            # 3) 사람의 결정을 가지고 정지된 지점부터 이어서 실행
            result = agent.invoke(Command(resume=resume_payload), config=config)

        # 4) 최종 AI 답변 출력
        final_message = result["messages"][-1]
        if isinstance(final_message, AIMessage):
            print(f"AI: {final_message.content}\n")
        else:
            print(f"AI: {final_message.content}\n")


if __name__ == "__main__":
    run_interactive()