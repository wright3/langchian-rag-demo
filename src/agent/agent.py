import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from src.prompt.agent_prompts import get_prompt_template
from src.tools.search import search_docs
from src.utils.middleware import LoggingMiddleware, MessageTrimmerMiddleware

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = init_chat_model(
    "groq:llama-3.3-70b-versatile",  # Groq 提供的 Llama 3.3 模型
    api_key=GROQ_API_KEY
)

agent = create_agent(
    model,
    tools=[search_docs],
    middleware=[LoggingMiddleware(), MessageTrimmerMiddleware(max_messages=4)]
)

prompt = get_prompt_template
chain = prompt | agent
