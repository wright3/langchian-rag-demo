# coding:utf-8
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor

load_dotenv()
ZAI_API_KEY = os.getenv("ZAI_API_KEY")

model = ChatOpenAI(
    temperature=0.6,
    model="GLM-4.5-Flash",
    openai_api_key=ZAI_API_KEY,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)


# Create specialized agents

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


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


math_agent = create_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    system_prompt="You are a math expert. Always use one tool at a time."
)

research_agent = create_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    system_prompt="You are a world class researcher with access to web search. Do not do any math."
)

# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    )
)

# Compile and run
app = workflow.compile()

for chunk in app.stream({
    "messages": [
        {
            "role": "user",
            "content": "算一下3+4*5，并告诉我明天深圳天气怎么样？"
        }
    ]
}):
    print(chunk)
