from functools import lru_cache

from langchain_mcp_adapters.client import MultiServerMCPClient


@lru_cache
async def load_deepwiki_tools():
    client = MultiServerMCPClient({'deepwiki': {'url': 'https://mcp.deepwiki.com/mcp', 'transport': 'streamable_http'}})
    tools = await client.get_tools()

    return tools
