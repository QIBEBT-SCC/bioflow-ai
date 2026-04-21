from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command, interrupt

from llm.states import UserContext


@tool('ask_human', parse_docstring=True)
def ask_human(question: str) -> str:
    """Consult humans for advice.

    Args:
        question: Issues to consult with humans
    """
    answer = interrupt({'question': question})
    return f'Human feedback is: "{answer}"'


@tool('ask_human')
async def ask_human_and_end(question: str, runtime: ToolRuntime) -> Command:
    """Seek human feedback."""
    # 若自动运行模式开启, 直接批准
    if getattr(runtime.context, 'auto_approve', False):
        return Command(
            update={
                'should_end': True,
                'updated': False,
                'messages': [
                    ToolMessage(content='The user agreed with your point.', tool_call_id=runtime.tool_call_id)
                ],
            },
        )

    answer = interrupt({'question': question})

    if isinstance(answer, bool) and answer:
        return Command(
            update={
                'should_end': True,
                'updated': False,
                'messages': [
                    ToolMessage(content='The user agreed with your point.', tool_call_id=runtime.tool_call_id)
                ],
            },
        )

    return Command(
        update={
            'should_end': False,
            'updated': False,
            'messages': [ToolMessage(content=f'Human feedback is: "{answer}"', tool_call_id=runtime.tool_call_id)],
        },
    )
