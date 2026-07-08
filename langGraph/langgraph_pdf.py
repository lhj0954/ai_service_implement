from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

# ------------------------------
# 1. 벡터스토어 & 리트리버 준비
# ------------------------------
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding,
    collection_name="pdf_collection"
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# 마지막 검색 결과를 저장해둘 전역 변수 (표준출력용)
_last_docs = []


# ------------------------------
# 2. RAG 검색 Tool 정의
# ------------------------------
@tool
def rag_search(query: str) -> str:
    """PDF 문서(LangGraph 관련 자료)에서 사용자의 질문과 관련된 내용을 검색합니다.
    문서 기반 사실 확인이나 구체적인 내용이 필요한 질문일 때 반드시 이 도구를 사용하세요.
    """
    global _last_docs
    docs = retriever.invoke(query)
    _last_docs = docs

    context = "\n\n".join(doc.page_content for doc in docs)
    return context


# ------------------------------
# 3. LLM + Agent 구성
# ------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

system_prompt = (
    "당신은 PDF 문서 분석 전문가입니다. "
    "사용자의 질문에 답하기 위해 필요하면 rag_search 도구를 사용해 "
    "문서 내용을 검색한 뒤, 검색된 내용을 근거로 답변하세요. "
    "문서에서 답을 찾을 수 없으면 모른다고 답하세요."
)

agent = create_react_agent(
    model=llm,
    tools=[rag_search],
    prompt=system_prompt
)


# ------------------------------
# 4. 검색 문서 출력 함수
# ------------------------------
def print_docs(docs):
    if not docs:
        print("\n(이번 답변에서는 문서 검색을 사용하지 않았습니다)")
        return

    print("\n========== 검색 문서 ==========\n")
    for i, doc in enumerate(docs, 1):
        print(f"[{i}]")
        page = doc.metadata.get("page", "?")
        print(f"페이지 : {page}")
        print(doc.page_content[:300])
        print("-" * 60)


# ------------------------------
# 5. 표준 입력 루프
# ------------------------------
while True:
    question = input("\n질문(q 종료): ")
    if question.lower() == "q":
        break

    _last_docs = []  # 매 질문마다 초기화

    result = agent.invoke(
        {"messages": [("user", question)]}
    )

    # 검색 결과 표준출력
    print_docs(_last_docs)

    # 최종 답변 출력
    print("\n========== AI 답변 ==========\n")
    final_message = result["messages"][-1]
    print(final_message.content)