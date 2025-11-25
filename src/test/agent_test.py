# coding:utf-8

from langgraph_sdk import get_sync_client  # or get_client for async

client = get_sync_client(url="http://127.0.0.1:2024")

for chunk in client.runs.stream(
        None,  # Threadless run
        "agent",  # Name of agent. Defined in langgraph.json.
        input={
            "messages": [{
                "role": "human",
                "content": "What is LangGraph?",
            }],
        },
        stream_mode="updates",
):
    print(f"Receiving new event of type: {chunk.event}...")
    print(chunk.data)
    print("\n\n")
