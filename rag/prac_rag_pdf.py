from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

loader = PyPDFLoader("sample.pdf")

documents = loader.load()

print("페이지 수 :", len(documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

split_docs = splitter.split_documents(documents)

print("분할 문서 :", len(split_docs))


embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory="./chroma_db",
    collection_name="pdf_collection"
)

print("Chroma 저장 완료")