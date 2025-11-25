# coding:utf-8

from langchain_core.prompts import ChatPromptTemplate


def get_prompt_template():
    template = ChatPromptTemplate.from_messages([
        ("system", """
        你是一个有用的助手，可以使用以下工具：
        - web_search: 用于搜索最新网络信息
        - search_knowledge_base: 用于搜索本地知识库
        - calculator: 用于计算
        
        请根据用户问题选择合适的工具。"""),
        ("user", "{input}")
    ])
    return template
