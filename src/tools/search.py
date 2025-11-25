# coding:utf-8

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.utils.rag import pinecorn_vectorstore, chroma_vectorstore


class SearchDocsInput(BaseModel):
    query: str = Field(description="需要搜索的文本")


@tool(args_schema=SearchDocsInput)
def search_knowledge_base(query: str) -> str:
    """Search for information in the knowledge base

    Args:
        query: Search query string
    """
    pinecorn = pinecorn_vectorstore()
    pinecorn_retriever = pinecorn.as_retriever(
        search_kwargs={"k": 3}
    )
    chroma = chroma_vectorstore()
    chroma_retriever = chroma.as_retriever(
        search_kwargs={"k": 3}
    )
    ensembel_retriever = EnsembleRetriever(
        retrievers=[pinecorn_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )

    docs = ensembel_retriever.invoke(query, k=4)
    return "\n\n".join([doc.page_content for doc in docs])


@tool(args_schema=SearchDocsInput)
def web_search(query: str) -> str:
    """Search for real-time information on the internet

    Args:
        query: Search query keywords
    """
    try:
        search = DuckDuckGoSearchRun(region="cn-zh")
        results = search.run(query)
        return results if results else "No relevant information found"
    except Exception as e:
        return f"Search failed: {str(e)}"


class CalculatorInput(BaseModel):
    operation: str = Field(description="运算类型add,subtract,multiply,divide")
    a: float = Field(description="第一个数字")
    b: float = Field(description="第二个数字")


@tool(args_schema=CalculatorInput)
def calculator(operation: str, a: float, b: float) -> str:
    """
    执行基本的数学计算

    参数:
        operation: 运算类型，支持 "add"(加), "subtract"(减), "multiply"(乘), "divide"(除)
        a: 第一个数字
        b: 第二个数字

    返回:
        计算结果字符串
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "错误：除数不能为零"
    }

    if operation not in operations:
        return f"不支持的运算类型：{operation}。支持的类型：add, subtract, multiply, divide"

    try:
        result = operations[operation](a, b)
        return f"{a} {operation} {b} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


if __name__ == '__main__':
    # print(web_search.invoke({"query": "深圳明天天气和温度怎么样？"}))

    print(search_knowledge_base.invoke({"query": "段誉是谁"}))
