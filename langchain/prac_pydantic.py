from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: str
    age: int | None = None # 선택적 필드 (기본값 None)

user = User(id=1, name="홍길동", email="hong@test.com")
print(user.id) # 1
print(user.model_dump()) # dict로 변환
print(user.model_dump_json()) # JSON 문자열로 변환

User(id="not_a_number", name="hong", email="x")