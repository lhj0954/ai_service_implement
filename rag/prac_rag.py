'''
Wikipedia(URL)  →  WebBaseLoader  →    Document  →   RecursiveCharacterTextSplitter
 → OpenAIEmbeddings  →  InMemoryVectorStore   →  Retriever   →   Prompt  → ChatOpenAI   →  답변
'''
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings
)

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


url = "https://ko.wikipedia.org/wiki/%EC%9C%A4%EB%8F%99%EC%A3%BC"
loader = WebBaseLoader(url)
documents = loader.load()
print("로드된 문서 수 :", len(documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
split_docs = splitter.split_documents(documents)
print("분할된 문서 수 :", len(split_docs))

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = InMemoryVectorStore.from_documents(
    split_docs,
    embedding
)

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4
    }
)

prompt = ChatPromptTemplate.from_template(
"""
당신은 윤동주 전문가입니다.
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
    question = input("\n질문 (종료:q) : ")
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

    print("\n========== 검색된 문서 ==========\n")

    for i, doc in enumerate(docs, 1):
        print(f"[{i}]")
        print(doc.page_content[:200])
        print()

    print("\n========== AI 답변 ==========\n")

    print(response.content)