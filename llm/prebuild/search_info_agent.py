from typing import Optional

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field

from llm.models import gemini_25_flash
from llm.tools.web_search import web_reader, web_searcher

SEARCH_SYS = """To organize information about bioinformatics tools, we need to search for their corresponding GitHub repositories and paper links. 
To search for this information, you need to call the `web_search` tool to search online for relevant information. 

For papers, you can use the following steps to query:
- First, try calling the `web_reader` tool to directly read the tool's GitHub page to see if there is a description of references on the homepage.
- If not, try searching online. If both preprints and published literature are found, prioritize published literature.
- If there are multiple relevant papers, prioritize the earliest published, original paper for that tool.
- If none are found, the paper link can be empty.

When you are sure you have enough information, do not call the tool again, return the result, and the task ends."""

SUMMARY_SYS = """You are an expert summarizer creating a concise log of previously completed web research to prevent redundant work.
Analyze the provided context of `web_searcher` and `web_reader` tool calls. Your task is to create a historical record of what has been searched and what was learned from the pages that were actually **read**.
Group the results under the specific search query that led to them.

**Strictly adhere to the following nested format for your output:**

*   **Query:** "[Insert the exact search query text here]"
    *   **Link:** [The URL that was read]
    *   **Extracted Information:** [A summary of the key information from that page. If the content was irrelevant or unhelpful, state "No useful information found."]

*If multiple pages were read for a single query, list each one as a separate "Link" entry under that query.*
"""


class OutputSchema(BaseModel):
    github_url: str = Field(description='GitHub repository link for the tool')
    paper_url: Optional[str] = Field(description="Link to the tool's publication, leave blank if not applicable")


response_format = ToolStrategy(OutputSchema)

search_info_agent = create_agent(
    model=gemini_25_flash(0.3),
    tools=[web_searcher, web_reader],
    system_prompt=SEARCH_SYS,
    middleware=[
        SummarizationMiddleware(
            model=gemini_25_flash(), max_tokens_before_summary=6000, messages_to_keep=6, summary_prompt=SUMMARY_SYS
        )
    ],
    response_format=ToolStrategy(OutputSchema),
)
