from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from app.utils.file.storage import read_text_file


@tool(parse_docstring=True)
async def view_pdf(file_id: str, runtime: ToolRuntime):
    """Request to obtain the full text of the PDF file corresponding to the ID

    Args:
        file_id: The ID of the file in the storage system, such as `documents/75/f5/4c/s41467-022-35735-y.pdf`
    """

    try:
        context = read_text_file(file_id.replace('.pdf', '.md'))
        context = f'<Doc>\n{context}\n</Doc>'
    except Exception:
        context = f'Error reading file: "{file_id}", please verify that the file name is correct.'

    return Command(
        update={
            'paper_id': file_id,
            'messages': [ToolMessage(content=context, tool_call_id=runtime.tool_call_id)],
        }
    )
