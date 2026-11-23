"""Build reusable structured-output runnables for LLM calls."""

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

from bioflow_ai.llm.context import BaseContext
from bioflow_ai.llm.llm_client import LLMMetadata
from bioflow_ai.llm.middleware import EndLoopMiddleware, RequireToolCallMiddleware, TokenStatisticMiddleware
from bioflow_ai.utils.llm.artifacts import latest_tool_artifact

if TYPE_CHECKING:
    from langchain.agents.middleware import AgentMiddleware

FORCE_TOOL_PROMPT = """You must produce the final answer by calling `submit_structured_output` exactly once.
Do not answer in natural language or perform any other action."""


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
                'should_end': True,
                'messages': [
                    ToolMessage(
                        content='Structured output has been captured.',
                        tool_call_id=runtime.tool_call_id,
                        artifact={'structured_output': parsed},
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


def _get_target_structure[SchemaT: BaseModel](
    output: Mapping[str, Any],
    schema: type[SchemaT],
) -> SchemaT:
    artifact = latest_tool_artifact(output.get('messages', ()), 'structured_output')
    if artifact is None:
        msg = 'Structured output agent finished without structured_output artifact.'
        raise ValueError(msg)
    raw = artifact['structured_output']
    return raw if isinstance(raw, schema) else schema.model_validate(raw)


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
    middleware: list[AgentMiddleware[AgentState[Any], BaseContext]] = [
        RequireToolCallMiddleware(
            'Before concluding the task, invoke the `submit_structured_output` tool with arguments matching the required schema.'
        ),
        EndLoopMiddleware(),
    ]
    if metadata:
        middleware.append(TokenStatisticMiddleware(name, **metadata))

    structure_agent = create_agent(
        llm,
        [_create_output_tool(schema)],
        system_prompt=prompt,
        middleware=middleware,
        context_schema=BaseContext,
    )

    def invoke(input_: LanguageModelInput, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        output = structure_agent.invoke(
            _coerce_messages(input_),
            config=config,
            context=kwargs.get('context'),
        )
        return _get_target_structure(output, schema)

    async def ainvoke(input_: LanguageModelInput, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        output = await structure_agent.ainvoke(
            _coerce_messages(input_),
            config=config,
            context=kwargs.get('context'),
        )
        return _get_target_structure(output, schema)

    return RunnableLambda(invoke, afunc=ainvoke, name=name)
