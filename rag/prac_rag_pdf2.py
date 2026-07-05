from dotenv import load_dotenv
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings
)
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding,
    collection_name="pdf_collection"
)


retriever = vectorstore.as_retriever(
    search_kwargs={
        "k":4
    }
)

prompt = ChatPromptTemplate.from_template(
"""
당신은 PDF 문서 분석 전문가입니다.
아래 문서를 참고하여 질문에 답변하세요.
=========================
{context}
=========================
질문
{question}
"""
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

while True:
    question = input("\n질문(q 종료): ")
    if question.lower() == "q":
        break

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    messages = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    response = llm.invoke(messages)

    print("\n========== 검색 문서 ==========\n")
    for i, doc in enumerate(docs, 1):
        print(f"[{i}]")
        page = doc.metadata.get("page", "?")
        print(f"페이지 : {page}")
        print(doc.page_content[:300])
        print("-"*60)

    print("\n========== AI 답변 ==========\n")
    print(response.content)