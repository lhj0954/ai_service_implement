import os
import json
from typing import Any, Callable, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

class OutputParserException(Exception):
    """파싱 실패 시 발생시키는 커스텀 예외"""
    pass


T = TypeVar("T", bound=BaseModel)

class PydanticOutputParser:
    """
    LLM이 반환한 텍스트를 지정된 Pydantic 모델로 파싱하는 파서.

    - JSON 문자열 -> dict -> Pydantic 모델 순으로 변환
    - JSON 자체가 깨졌거나(JSONDecodeError), 필드 타입/필수값이
      틀린 경우(ValidationError) 모두 OutputParserException으로 통일해서 던짐
    """

    def __init__(self, pydantic_model: Type[T]):
        self.pydantic_model = pydantic_model

    def get_format_instructions(self) -> str:
        # 모델의 JSON 스키마를 그대로 포맷 지시문에 포함시켜
        # LLM이 정확한 필드명/타입을 참고하도록 함
        schema = self.pydantic_model.model_json_schema()
        return (
            "출력은 반드시 아래 JSON 스키마를 만족하는 JSON 객체 하나여야 합니다.\n"
            "다른 설명이나 텍스트 없이 JSON 객체만 출력하세요.\n\n"
            f"JSON 스키마:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
        )

    def parse(self, text: str) -> T:
        # 1) 문자열 -> dict
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise OutputParserException(
                f"JSON 파싱 실패: {e}\n원본 텍스트: {text!r}"
            )

        # 2) dict -> Pydantic 모델 (여기서 타입/필수 필드 검증까지 수행)
        try:
            return self.pydantic_model.model_validate(data)
        except ValidationError as e:
            raise OutputParserException(
                f"스키마 검증 실패: {e}\n원본 데이터: {data!r}"
            )


# ---------------------------------------------------------
# 2. OutputFixingParser 핵심 로직  
# ---------------------------------------------------------
class OutputFixingParser:
    """
    base_parser로 파싱을 시도하고, 실패하면 LLM(llm_call)에게
    수정을 요청하여 재파싱하는 래퍼 파서.
    """

    FIX_PROMPT_TEMPLATE = """다음 텍스트는 지정된 포맷에 맞지 않아 파싱에 실패했습니다.

--- 포맷 지시문 ---
{format_instructions}

--- 원본 출력(잘못된 형식) ---
{completion}

--- 발생한 에러 ---
{error}

위 에러를 참고하여, 포맷 지시문에 맞는 올바른 결과만 출력하세요.
설명이나 다른 텍스트 없이 수정된 JSON 결과만 반환하세요.
"""

    def __init__(
        self,
        base_parser: PydanticOutputParser,
        llm_call: Callable[[str], str],
        max_retries: int = 3,
    ):
        self.base_parser = base_parser
        self.llm_call = llm_call
        self.max_retries = max_retries

    def _build_fix_prompt(self, completion: str, error: Exception) -> str:
        return self.FIX_PROMPT_TEMPLATE.format(
            format_instructions=self.base_parser.get_format_instructions(),
            completion=completion,
            error=str(error),
        )

    def parse(self, completion: str) -> Any:
        current_output = completion
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                result = self.base_parser.parse(current_output)
                if attempt > 0:
                    print(f"[성공] {attempt}번째 수정 시도에서 파싱 성공")
                return result

            except OutputParserException as e:
                last_error = e
                print(f"[시도 {attempt}] 파싱 실패: {e}")

                if attempt >= self.max_retries:
                    break

                fix_prompt = self._build_fix_prompt(current_output, e)
                print(f"[시도 {attempt}] ChatGPT API에 수정 요청 전송...")
                current_output = self.llm_call(fix_prompt)

        raise OutputParserException(
            f"{self.max_retries}번의 재시도 후에도 파싱 실패. 마지막 에러: {last_error}"
        )


# ---------------------------------------------------------
# 3. ChatGPT API 호출 함수
# ---------------------------------------------------------
class ChatGPTCaller:

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        # api_key를 명시하지 않으면 환경변수 OPENAI_API_KEY를 자동으로 사용
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature

    def __call__(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": "너는 JSON 포맷 오류를 정확히 고치는 어시스턴트야. "
                                "설명 없이 수정된 JSON만 출력해.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    # (1) 원하는 출력 스키마를 Pydantic 모델로 정의
    class Person(BaseModel):
        name: str
        age: int

    # Pydantic 기반 파서 생성
    base_parser = PydanticOutputParser(pydantic_model=Person)

    llm_call = ChatGPTCaller(model="gpt-4o-mini", temperature=0.0)

    #OutputFixingParser 조립
    fixing_parser = OutputFixingParser(
        base_parser=base_parser,
        llm_call=llm_call,
        max_retries=2,
    )

    # 일부러 형식이 깨진 LLM 출력 예시  - 작은따옴표 사용 (JSON 파싱 자체가 실패)
    broken_output = "{'name': '홍길동', 'age': 30}"

    #  (JSON 파싱은 되지만 Pydantic 검증 실패)
    # broken_output = '{"name": "홍길동", "age": "삼십"}'

    print("=== OutputFixingParser (ChatGPT + Pydantic) 실행 ===")
    result: Person = fixing_parser.parse(broken_output)

    print("\n최종 파싱 결과:", result)
    print("타입:", type(result))
    print("name 필드:", result.name)
    print("age 필드:", result.age, type(result.age))