import base64
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# 이미지 파일을 Base64로 인코딩
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

image_data1 = encode_image("./image1.jpg.")
image_data2 = encode_image("./image2.jpg")

model = init_chat_model("gpt-4o")
#여러 이미지 동시 처리
message = HumanMessage(
    content=[
        {"type": "text", "text": "두 이미지를 비교해주세요."},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data1}"}
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data2}"}
        },
    ]
)

response = model.invoke([message])
print(response.content)