import json
import os
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage
)
 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
load_dotenv()

class JsonFileChatMessageHistory(BaseChatMessageHistory):

    def __init__(self, session_id: str, directory="history"):
        self.session_id = session_id
        self.directory = directory

        os.makedirs(directory, exist_ok=True)

        self.file_path = os.path.join(
            directory,
            f"{session_id}.json"
        )

    @property
    def messages(self):

        if not os.path.exists(self.file_path):
            return []

        with open(
            self.file_path,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        messages = []

        for item in data:

            if item["type"] == "human":
                messages.append(
                    HumanMessage(content=item["content"])
                )

            elif item["type"] == "ai":
                messages.append(
                    AIMessage(content=item["content"])
                )

        return messages

    def add_message(self, message: BaseMessage):

        data = []

        if os.path.exists(self.file_path):

            with open(
                self.file_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

        if isinstance(message, HumanMessage):

            data.append({
                "type": "human",
                "content": message.content
            })

        elif isinstance(message, AIMessage):

            data.append({
                "type": "ai",
                "content": message.content
            })

        with open(
            self.file_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=4
            )

    def clear(self):

        if os.path.exists(self.file_path):
            os.remove(self.file_path)

def get_session_history(session_id: str):
    return JsonFileChatMessageHistory(session_id)


llm = ChatOpenAI(    model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 친절한 AI 비서입니다."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ]
)

chain = prompt | llm

chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


config = {
    "configurable": {
        "session_id": "conversation_1"
    }
}


response = chatbot.invoke(
    {
        "question": "안녕하세요"
    },
    config=config
)

print(response.content)


response = chatbot.invoke(
    {
        "question": "내 이름은 이학준입니다."
    },
    config=config
)

print(response.content)


response = chatbot.invoke(
    {
        "question": "내 이름이 뭐였죠?"
    },
    config=config
)

print(response.content)