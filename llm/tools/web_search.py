from typing import Optional

from crawl4ai import AsyncWebCrawler, BM25ContentFilter, BrowserConfig, CrawlerRunConfig, DefaultMarkdownGenerator
from ddgs import DDGS
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tools import tool

from app.core.config import get_settings
from llm.models import get_long_context_model

_SUMMARY_SYS = """# Role: Context Compression Engine
You are an AI assistant specializing in compressing conversation histories. Your core mission is to precisely and efficiently distill key information from long dialogues, generating a structured summary that is easy for a subsequent AI to understand and use.

Your primary objectives are:
1.  Preserve the user's core intent and task progress.
2.  Log all executed web search (`web_searcher`) and web reading (`web_reader`) operations to inform future decisions and **avoid redundant work**.
3.  Your output format must strictly adhere to the Markdown and XML specifications below.


# Compression Instructions
You will be given a complete conversation history. Compress it according to the following steps:

1.  **Identify the Core Task**: Analyze the entire conversation to summarize the user's primary, fundamental task or question.
2.  **Distill Key Information**: Extract confirmed facts, critical data, important conclusions, or intermediate results from the conversation. All valuable information obtained from web reading (`web_reader`) should be summarized here.
3.  **Track Tool Usage**: This is the most critical part. Carefully scan the conversation history for all `web_searcher` and `web_reader` tool calls and log only their invocation parameters.
    *   For `web_searcher`, extract the search queries used.
    *   For `web_reader`, extract only the URLs of the web pages that have been read.


# Output Format
You must strictly organize your compressed output using the following Markdown headers and XML block structure.

```markdown
# Core Task / User Intent
(Clearly summarize the user's ultimate goal in one or two sentences here.)

# Key Information & Current Progress
*   (Use a bulleted list here to summarize established key facts, data, or conclusions. This section should include all core information distilled from the web pages that have been read.)
*   (e.g., Confirmed that Product A's release date is October 2023.)
*   (e.g., According to the official website, Product A's technical specs include...)
*   (e.g., Currently comparing the performance parameters of Product A and Product B.)

# Tool Usage History
<tool_history>
  <searches>
    <!-- Log all search queries that have been used -->
    <query>The first search query string used</query>
    <query>The second search query string used</query>
  </searches>
  <reads>
    <!-- Log only the URLs of web pages that have been read to avoid re-reading -->
    <url source="https://example.com/page1"/>
    <url source="https://example.com/page2"/>
  </reads>
</tool_history>
```
"""


@tool(parse_docstring=True)
async def web_searcher(query: str) -> str:
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


@tool(parse_docstring=True)
async def web_reader(url: str, query: Optional[str] = None) -> str:
    """Calling this tool allows you to read the content of a specific web page.

    **This tool prohibits parallel calls! You are only allowed to read one webpage at a time!**

    Args:
        url: URL of the web page to read
        query: If this parameter is used, the page content will be filtered by BM25 using the provided query; otherwise, the original page is returned.
    """
    proxy = {'server': get_settings().network.get_proxy_url()} if get_settings().network.USE_PROXY else None

    async with AsyncWebCrawler(config=BrowserConfig(proxy_config=proxy, verbose=False, text_mode=True)) as crawler:
        if query:
            bm25_filter = BM25ContentFilter(
                user_query=query,
                # Adjust for stricter or looser results
                bm25_threshold=1.0,
            )
            md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)
            run_config = CrawlerRunConfig(verbose=False, markdown_generator=md_generator)
            result = await crawler.arun(url=url, config=run_config)
            return f'<PageContent>\n{result.markdown.fit_markdown}\n</PageContent>\n\n'

        run_config = CrawlerRunConfig(verbose=False, exclude_all_images=True)
        result = await crawler.arun(url=url, config=run_config)

        if result.markdown and len(result.markdown) > 8000:  # noqa: PLR2004
            return f'<PageContent>\n{result.markdown[:8000]}...\n</PageContent>\n\n'
        return f'<PageContent>\n{result.markdown}\n</PageContent>\n\n'


web_summary_middleware = SummarizationMiddleware(
    model=get_long_context_model(), max_tokens_before_summary=10000, messages_to_keep=4, summary_prompt=_SUMMARY_SYS
)
