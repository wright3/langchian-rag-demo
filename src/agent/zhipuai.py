import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.tools.search import calculator, search_knowledge_base, web_search
from src.utils.middleware import LoggingMiddleware, MessageTrimmerMiddleware, MaxCallsMiddleware

load_dotenv()
ZAI_API_KEY = os.getenv("ZAI_API_KEY")

model = ChatOpenAI(
    temperature=0.6,
    model="GLM-4.5-Flash",
    openai_api_key=ZAI_API_KEY,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# 智谱AI
agent = create_agent(
    model,
    tools=[calculator, search_knowledge_base, web_search],
    middleware=[
        LoggingMiddleware(),
        MessageTrimmerMiddleware(max_messages=4),
        MaxCallsMiddleware(max_calls=10),
    ],
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{domain}专家"),
    ("human", "{topic}")
])
chain = prompt | agent

print(chain.invoke({'domain': '智能', 'topic': '你好'}))
