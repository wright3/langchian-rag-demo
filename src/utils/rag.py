# coding:utf-8
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

ROOTPATH = Path(__file__).parent.parent

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


def read_doc():
    """文本读取"""
    doc_path = ROOTPATH / "doc/role.txt"
    # 使用 TextLoader 加载
    loader = TextLoader(doc_path, encoding="utf-8")
    documents = loader.load()
    return documents


def spilt_doc(documents):
    """文本分割"""
    # 创建文本分割器
    separators = ["\n\n", "\n", " ", ""]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50, length_function=len,
                                              separators=separators)

    # 分割文档
    chunks = splitter.split_documents(documents)
    return chunks


def pinecorn_embed_doc(chunks):
    """向量化"""
    # 创建向量化模型
    embeddings = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    vector = embeddings.embed_query("天龙八部人物身份详解")

    # 初始化 Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "langchain-rag-tlbb"
    dimension = len(vector)
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",  # 相似度度量
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # 免费层级可用区域
            )
        )
        index = pc.Index(index_name)
    else:
        index = pc.Index(index_name)

    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    return vectorstore


def pinecorn_vectorstore():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "langchain-rag-tlbb"
    index = pc.Index(index_name)
    embeddings = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    return vector_store


def bm25_retriever(chunks):
    """BM25 检索器"""
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3
    return bm25_retriever


def chroma_embed_doc(chunks):
    """chroma向量化"""

    ids = [f"tlbb_{i}" for i in range(len(chunks))]

    vectorstore = Chroma(
        collection_name="langchain-rag-tlbb",
        embedding_function=HuggingFaceEmbeddings(model_name="moka-ai/m3e-base"),
        persist_directory=ROOTPATH / "doc/chroma_tlbb",
    )

    vectorstore.add_documents(
        documents=chunks,
        ids=ids
    )

    return vectorstore


def chroma_vectorstore():
    """chroma检索"""
    vectorstore = Chroma(
        collection_name="langchain-rag-tlbb",
        embedding_function=HuggingFaceEmbeddings(model_name="moka-ai/m3e-base"),
        persist_directory=ROOTPATH / "doc/chroma_tlbb",
    )
    return vectorstore


if __name__ == '__main__':
    doc = read_doc()
    chunks = spilt_doc(doc)
    vectorstore = pinecorn_embed_doc(chunks)
    print('done')
