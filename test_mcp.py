"""
Test the AI Database Assistant MCP server by spawning it as a subprocess
and calling the ask_database tool over stdio.
"""
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

load_dotenv()

# Project root (where .env and ai_database_assistant.py live)
PROJECT_ROOT = Path(__file__).resolve().parent


async def main():
    transport = StdioTransport(
        command="python",
        args=["ai_database_assistant.py"],
        cwd=PROJECT_ROOT,
        env={**os.environ},
        keep_alive=True,
    )
    client = Client(transport)
    loop = asyncio.get_event_loop()

    async with client:
        tools = await client.list_tools()
        print("Tools:", [t.name for t in tools])
        print("Ask questions about the database (or type 'exit' to quit).\n")

        while True:
            question = await loop.run_in_executor(
                None, lambda: input("Question: ").strip()
            )
            if not question or question.lower() == "exit":
                break

            result = await client.call_tool("ask_database", {"question": question})
            text = result.content[0].text if result.content else str(result)
            print("\n", text, "\n")


if __name__ == "__main__":
    asyncio.run(main())
