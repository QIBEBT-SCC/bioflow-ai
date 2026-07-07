from typing import Any, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT, hook_config
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from langgraph.typing import ContextT


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


class EndLoopSchema(AgentState):
    should_end: NotRequired[bool]


class AskAndEndMiddleware(AgentMiddleware):
    tools = [ask_human_and_end]  # noqa: RUF012
    state_schema = EndLoopSchema

    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: D102
        return {'should_end': False}

    @hook_config(can_jump_to=['end'])
    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: D102
        if state.get('should_end', False):
            return {'jump_to': 'end'}

        return None

    @hook_config(can_jump_to=['end'])
    async def abefore_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: D102
        return self.before_model(state, runtime)


class EndLoopMiddleware(AgentMiddleware):
    state_schema = EndLoopSchema

    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: D102
        return {'should_end': False}

    @hook_config(can_jump_to=['end'])
    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: D102
        if state.get('should_end', False):
            return {'jump_to': 'end'}

        return None

    @hook_config(can_jump_to=['end'])
    async def abefore_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: D102
        return self.before_model(state, runtime)
