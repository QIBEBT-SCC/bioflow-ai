from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from loguru import logger

from bioflow_ai.llm.context import BaseContext
from bioflow_ai.services.documents.pdf_index import read_pdf


@tool(parse_docstring=True)
async def view_pdf(file_id: str, runtime: ToolRuntime[BaseContext], query: str | None = None) -> Command:
    """Read an attached PDF as full text or relevant paragraphs in original order.

    Omit query to read the complete Markdown. Provide query to retain paragraphs
    exceeding the system's similarity threshold, with omissions marked in place.
    PDF tags in conversation history only identify attachments; call this tool
    before claiming to have read their content.

    Args:
        file_id: The exact S3 ID supplied by the user's PDF tag.
        query: Optional question or topic for paragraph-level semantic filtering.
    """
    try:
        context = await read_pdf(file_id, query)
    except Exception as e:
        logger.error(e)
        context = f'Error reading PDF: {e}'

    return Command(
        update={
            'paper_id': file_id,
            'messages': [ToolMessage(content=context, tool_call_id=runtime.tool_call_id)],
        }
    )
