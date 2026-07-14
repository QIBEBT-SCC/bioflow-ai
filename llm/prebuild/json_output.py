from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from langchain.agents import AgentState, create_agent
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel

from llm.context import BaseContext
from llm.llm_client import LLMMetadata
from llm.utils.middleware import CheckStateExistMiddleware, EndLoopMiddleware, TokenStatisticMiddleware

if TYPE_CHECKING:
    from langchain.agents.middleware import AgentMiddleware

FORCE_TOOL_PROMPT = """You must produce the final answer by calling `submit_structured_output` exactly once.
Do not answer in natural language or perform any other action."""


class JsonOutputState(AgentState, total=False):
    target_structure: BaseModel
    should_end: bool
    updated: bool


def _tool_call_failed_message(error: Exception) -> str:
    return (
        'The structured output tool call failed validation. '
        f'Call the tool again with arguments that match the schema. Error: {error}'
    )


def _create_output_tool[SchemaT: BaseModel](schema: type[SchemaT]) -> StructuredTool:
    def build_command(runtime: ToolRuntime, **kwargs: Any) -> Command:
        parsed = schema.model_validate(kwargs)
        return Command(
            update={
                'target_structure': parsed,
                'updated': True,
                'should_end': True,
                'messages': [
                    ToolMessage(
                        content='Structured output has been captured.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    def submit_structured_output(runtime: ToolRuntime, **kwargs: Any) -> Command:
        return build_command(runtime, **kwargs)

    async def asubmit_structured_output(runtime: ToolRuntime, **kwargs: Any) -> Command:
        return build_command(runtime, **kwargs)

    return StructuredTool.from_function(
        submit_structured_output,
        coroutine=asubmit_structured_output,
        name='submit_structured_output',
        description='Return the final answer using the required structured schema.',
        args_schema=schema,
        handle_validation_error=_tool_call_failed_message,
    )


def _coerce_messages(input_: LanguageModelInput | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(input_, Mapping):
        if 'messages' in input_:
            return dict(input_)
        msg = 'json_output expects messages, a prompt value, or a string after prompt rendering.'
        raise TypeError(msg)

    if isinstance(input_, str):
        return {'messages': [HumanMessage(content=input_)]}

    if isinstance(input_, BaseMessage):
        return {'messages': [input_]}

    if isinstance(input_, PromptValue):
        return {'messages': input_.to_messages()}

    return {'messages': list(cast('list[BaseMessage]', input_))}


def _get_target_structure[SchemaT: BaseModel](output: Mapping[str, Any]) -> SchemaT:
    target_structure = output.get('target_structure')
    if target_structure is None:
        msg = 'Structured output agent finished without target_structure output.'
        raise ValueError(msg)
    return cast('SchemaT', target_structure)


def json_output[SchemaT: BaseModel](
    llm: BaseChatModel,
    system_prompt: str,
    schema: type[SchemaT],
    *,
    metadata: LLMMetadata | None = None,
    name: str = 'json_output',
) -> Runnable[LanguageModelInput, SchemaT]:
    """Create a structured-output runnable backed by a forced tool-call agent."""
    prompt = f'{system_prompt}\n\n{FORCE_TOOL_PROMPT}'
    middleware: list[AgentMiddleware] = [
        CheckStateExistMiddleware('target_structure', 'submit_structured_output'),
        EndLoopMiddleware(),
    ]
    if metadata:
        middleware.append(TokenStatisticMiddleware(name, **metadata))

    # pyrefly: ignore [no-matching-overload]
    structure_agent = create_agent(
        llm,
        [_create_output_tool(schema)],
        system_prompt=prompt,
        middleware=middleware,
        state_schema=JsonOutputState,
        context_schema=BaseContext,
    )

    def invoke(input_: LanguageModelInput, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        output = structure_agent.invoke(
            _coerce_messages(input_),
            config=config,
            context=kwargs.get('context'),
        )
        return _get_target_structure(output)

    async def ainvoke(input_: LanguageModelInput, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        output = await structure_agent.ainvoke(
            _coerce_messages(input_),
            config=config,
            context=kwargs.get('context'),
        )
        return _get_target_structure(output)

    return RunnableLambda(invoke, afunc=ainvoke, name=name)
