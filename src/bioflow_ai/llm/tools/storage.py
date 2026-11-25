from langchain_core.tools import tool
from loguru import logger

from bioflow_ai.services.documents.pdf_index import read_pdf


@tool(parse_docstring=True)
async def view_pdf(file_id: str, query: str | None = None) -> str:
    """Read an attached PDF as full Markdown or relevant paragraphs in original order.

    Omit `query` for full text, or provide `query` for paragraph-level semantic filtering with omissions marked.

    Args:
        file_id: The exact `id` attribute from the user's `<PDF id="...">` attachment tag.
        query: Optional question or topic for paragraph-level semantic filtering.
    """
    try:
        context = await read_pdf(file_id, query)
    except Exception as e:
        logger.error(e)
        context = f'Error reading PDF: {e}'

    return context
