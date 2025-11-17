# coding:utf-8

from langchain_core.tools import tool
from src.utils.rag import pinecorn_vectorstore


@tool
def search_docs(query: str) -> str:
    """在知识库中搜索相关信息"""
    vectorstore = pinecorn_vectorstore()
    docs = vectorstore.similarity_search(query, k=4)
    return "\n\n".join([doc.page_content for doc in docs])
