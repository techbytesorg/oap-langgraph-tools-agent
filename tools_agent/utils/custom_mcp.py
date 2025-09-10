import os
from typing import List
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


async def get_custom_mcp_tools() -> List[BaseTool]:
    """
    Load tools from a custom MCP server configured via environment variable.
    
    Returns:
        List of LangChain tools from the custom MCP server, or empty list if
        CUSTOM_MCP_URL is not set or if there are any errors.
    """
    custom_url = os.getenv("CUSTOM_MCP_URL")
    if not custom_url:
        return []
    
    client = MultiServerMCPClient({
        "custom": {
            "transport": "streamable_http",
            "url": custom_url,
        }
    })
    
    try:
        tools = await client.get_tools()
        print(f"Loaded {len(tools)} tools from custom MCP server at {custom_url}")
        return tools
    except Exception as e:
        print(f"Failed to load custom MCP tools from {custom_url}: {e}")
        return []