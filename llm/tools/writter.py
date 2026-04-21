from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command


@tool
async def write_report(content: str, runtime: ToolRuntime) -> Command:
    """Write or update tool report."""
    return_msg = (
        'Updated the report content.' if runtime.state.get('tool_report') else 'The report content has been saved.'
    )

    return Command(
        update={
            'tool_report': content,
            'updated': True,
            'messages': [ToolMessage(content=return_msg, tool_call_id=runtime.tool_call_id)],
        },
    )


@tool
async def write_snippet(content: str, runtime: ToolRuntime) -> Command:
    """Write the segment relevant to user requirements in the literature."""
    return Command(
        update={
            'paper_context': content,
            'updated': True,
            'messages': [ToolMessage(content='The paper context has been saved.', tool_call_id=runtime.tool_call_id)],
        },
    )


@tool
async def write_plan(content: str, runtime: ToolRuntime) -> Command:
    """Write or update plan."""
    return_msg = 'Updated the plan content.' if runtime.state.get('plan') else 'The plan content has been saved.'

    return Command(
        update={
            'plan': content,
            'updated': True,
            'messages': [ToolMessage(content=return_msg, tool_call_id=runtime.tool_call_id)],
        },
    )
