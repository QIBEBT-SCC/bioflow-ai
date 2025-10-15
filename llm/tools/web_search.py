import uuid
from typing import Annotated

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from ddgs import DDGS
from langchain_core.messages import RemoveMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentStatePydantic
from langgraph.types import Command

from app.core.config import get_settings


@tool
async def web_searcher(query: str):
    """Search the network based on keywords and return a series of web pages found.

    Args:
        query: Search query
    """
    proxy = get_settings().network.get_proxy_url() if get_settings().network.USE_PROXY else None
    with DDGS(proxy=proxy) as ddgs:
        search_result = ddgs.text(query, region='en', max_results=10)

    response_block = [
        (
            f'<result>\n'
            f'  <title>{d["title"]}</title>\n'
            f'  <url>{d["href"]}</url>\n'
            f'  <fragment>{d["body"]}</fragment>\n'
            f'</result>'
        )
        for d in search_result
    ]

    return '\n\n'.join(response_block)


@tool
async def web_reader(url: str):
    """Calling this tool allows you to read the content of a specific web page.

    Args:
        url: URL of the web page to read
    """
    proxy = get_settings().network.get_proxy_url() if get_settings().network.USE_PROXY else None
    async with AsyncWebCrawler(config=BrowserConfig(proxy=proxy, verbose=False, text_mode=True)) as crawler:
        result = await crawler.arun(url=url, config=CrawlerRunConfig(verbose=False, exclude_all_images=True))

    return f'<PageContent>\n{result.markdown}\n</PageContent>\n\n接下来，调用`conclude_web_reader_result`工具对此次阅读结果进行总结。'


@tool
async def conclude_web_reader_result(
    summary_content: str,
    state: Annotated[AgentStatePydantic, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Summarize the return result of `web_reader` tool

    Args:
        summary_content: The valid information contained in the last web page reading. If not, just say "There is no information needed in this web page."
    """
    tool_call_message = state.messages[-1].model_copy()
    tool_call_message.id = str(uuid.uuid4())
    new_messages = [
        ToolMessage(content='...', tool_call_id=state.messages[-2].tool_call_id),
        tool_call_message,
        ToolMessage(content='Web content has been summarized.', tool_call_id=tool_call_id),  # 简单的确认消息
    ]

    return Command(
        update={
            'messages': [
                RemoveMessage(id=state.messages[-2].id),
                RemoveMessage(id=state.messages[-1].id),
                *new_messages,
            ]
        }
    )
