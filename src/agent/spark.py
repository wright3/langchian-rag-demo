import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.tools.calculator import calculator
from src.tools.search import search_knowledge_base, web_search
from src.utils.middleware import LoggingMiddleware, MessageTrimmerMiddleware, MaxCallsMiddleware

load_dotenv()
IFLYTEK_SPARK_API_KEY = os.getenv("IFLYTEK_SPARK_API_KEY")
IFLYTEK_SPARK_API_URL = os.getenv("IFLYTEK_SPARK_API_URL")

# 讯飞
model = ChatOpenAI(
    temperature=0.3,
    model="xop3qwen1b7",
    openai_api_key=IFLYTEK_SPARK_API_KEY,
    openai_api_base=IFLYTEK_SPARK_API_URL
)
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
