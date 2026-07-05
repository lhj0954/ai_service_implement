'''
CSV 파일  →  CSVLoader  →  Document  →  RecursiveCharacterTextSplitter
 → OpenAIEmbeddings  →  FAISS   →  faiss_index (디스크 저장)  →  Retriever   →  Prompt
 →  ChatOpenAI  → 답변
'''
from dotenv import load_dotenv

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI
)

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

loader = CSVLoader(
    file_path="sample.csv",
    encoding="utf-8"
)

documents = loader.load()

print(f"문서 개수 : {len(documents)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

split_docs = splitter.split_documents(documents)

print(f"분할 문서 : {len(split_docs)}")

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = FAISS.from_documents(
    split_docs,
    embedding
)


vectorstore.save_local("faiss_index")

print("FAISS 저장 완료")


retriever = vectorstore.as_retriever(
    search_kwargs={
        "k":3
    }
)

prompt = ChatPromptTemplate.from_template(
"""
당신은 CSV 전문가입니다.
아래 문서를 참고하여 질문에 답변하세요.
==========================
{context}
==========================
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

    print("\n========== 검색 결과 ==========\n")

    for doc in docs:
        print(doc.page_content)
        print("-"*60)

    print("\n========== AI 답변 ==========\n")
    print(response.content)