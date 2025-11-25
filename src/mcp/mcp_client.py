import asyncio
import os.path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_qwq import ChatQwen

client = MultiServerMCPClient(
    {
        "math": {
            # Make sure you start your weather server on port 8000
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    }
)


async def main():
    load_dotenv()
    tools = await client.get_tools()
    print(tools)
    model = ChatQwen(
        model="qwen3-max",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    agent = create_agent(model, tools)
    math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
    print(math_response)


asyncio.run(main())
