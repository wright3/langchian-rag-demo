# coding:utf-8

import asyncio

from langgraph_sdk import get_client  # or get_client for async

client = get_client(url="http://127.0.0.1:2024")


async def main():
    async for chunk in client.runs.stream(
            None,  # Threadless run
            "agent",  # Name of agent. Defined in langgraph.json.
            input={
                "messages": [{
                    "role": "human",
                    "content": "你好",
                }],
            },
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")


asyncio.run(main())
