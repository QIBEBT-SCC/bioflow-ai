from pathlib import Path
from typing import Any, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT, hook_config
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from langgraph.typing import ContextT

from llm.context import ProjectContext


@tool('ask_human')
async def ask_human_and_end(question: str, runtime: ToolRuntime[ProjectContext]) -> Command:
    """Seek human feedback."""
    # 若自动运行模式开启, 直接批准
    if runtime.context.auto_approve:
        return Command(
            update={
                'should_end': True,
                'updated': False,
                'messages': [
                    ToolMessage(content='The user agreed with your point.', tool_call_id=runtime.tool_call_id)
                ],
            },
        )

    interrupt_payload = {'question': question}
    if runtime.state.get('plan') and runtime.state.get('updated'):
        interrupt_payload['kind'] = 'plan_approval'
        plan_file = Path(runtime.context.workdir) / 'PLAN.md'
        if plan_file.is_file():
            interrupt_payload['plan'] = plan_file.read_text(encoding='utf-8')
    answer = interrupt(interrupt_payload)

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

    @hook_config(can_jump_to=['model'])
    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: D102
        last_message = state['messages'][-1]
        if getattr(last_message, 'tool_calls', None):
            return None
        if state.get('updated') and not state.get('should_end', False):
            return {
                'jump_to': 'model',
                'messages': [
                    HumanMessage(
                        content=(
                            'The plan has been written but has not been confirmed. '
                            'Before concluding, call `ask_human` and ask the user to review and confirm the plan.'
                        )
                    )
                ],
            }
        return None

    @hook_config(can_jump_to=['model'])
    async def aafter_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: D102
        return self.after_model(state, runtime)


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
