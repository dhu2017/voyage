"""Voyage MCP Server — entry point."""

# Pre-import heavy C-extension libraries BEFORE the async event loop starts.
# On Windows, OpenBLAS/scipy/matplotlib initialise internal thread-pools and
# acquire locks during first import.  If that import happens inside an anyio
# worker thread while the IOCP proactor loop is running, the two lock
# hierarchies deadlock and the tool call never returns.
import numpy, scipy, pandas, riskfolio  # noqa: F401, E401

import json
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .tools.registry import TOOLS, list_tools


app = Server("voyage-invest")


@app.list_tools()
async def handle_list_tools() -> list[Tool]:
    """Return all registered tools."""
    return [
        Tool(
            name=t["name"],
            description=t["description"],
            inputSchema=t["inputSchema"],
        )
        for t in TOOLS.values()
    ]


@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch tool call to the registered handler."""
    tool = TOOLS.get(name)
    if tool is None:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False))]

    handler = tool["handler"]
    try:
        # Run handler synchronously in the event loop.
        # We pre-imported numpy/scipy/riskfolio at module level to avoid OpenBLAS
        # deadlock, so it's safe to call these functions directly without a thread pool.
        # Running sync avoids the anyio worker thread + Windows IOCP + OpenBLAS deadlock.
        result = handler(**arguments)
    except Exception as e:
        result = {"error": str(e)}

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, default=str))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())


# Allow `python -m voyage.server`
def cli():
    asyncio.run(main())
