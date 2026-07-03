from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = init_chat_model("gpt-4o-mini")

output_parser = CommaSeparatedListOutputParser()

# 포맷 지시사항 확인
format_instructions = output_parser.get_format_instructions()
print(format_instructions)

# Your response should be a list of comma separated values, eg: `foo, bar, baz`
# 프롬프트에 포맷 지시사항 포함
prompt = PromptTemplate(
    template="5가지 {subject}을(를) 나열하세요.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | llm | output_parser
result = chain.invoke({"subject": "청주에서 맛집"})
print(result)
print(type(result)) 