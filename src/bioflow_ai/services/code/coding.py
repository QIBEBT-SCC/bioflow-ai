"""Initialize CODE workspaces and run native coding agent turns."""

import difflib
import hashlib
import json
import tomllib
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol, cast

import tomli_w
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict

from bioflow_ai.core.config import get_settings
from bioflow_ai.llm.context import CodingContext
from bioflow_ai.llm.workflow_builder.foreach_manifest import foreach_manifest_contract
from bioflow_ai.schemas.code import CodeNodeType
from bioflow_ai.schemas.code_agent import CodeAgentProposal
from bioflow_ai.schemas.react_flow import WorkflowDefinition
from bioflow_ai.schemas.workflow_build import BuildEdge, BuildNode
from bioflow_ai.services.code.workspace import SCRIPT_FILENAMES, initialize_code_workspace_async
from bioflow_ai.tasks.code import normalize_python_dependencies
from bioflow_ai.utils.formatters import format_code_inputs
from bioflow_ai.utils.observability.execution_logging import AgentRunLogger

ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


class CodingAgent(Protocol):
    def astream_events(self, *args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        """Yield native model and tool events, ending with the root graph result."""
        ...

    async def ainvoke(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Run the agent to its final state."""
        ...


class CodingTurnResult(BaseModel):
    """Only a complete turn can update conversation history and offer a candidate."""

    model_config = ConfigDict(frozen=True)

    messages: list[BaseMessage]
    proposal: CodeAgentProposal | None


def message_text(message: Any) -> str:
    """Extract visible text from message objects or provider content blocks."""
    content = getattr(message, 'content', message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(
            item if isinstance(item, str) else str(item.get('text', ''))
            for item in content
            if isinstance(item, (str, dict))
        )
    return ''


def build_code_inputs(
    input_edges: list[BuildEdge],
    exist_nodes: list[BuildNode],
    exist_edges: list[BuildEdge],
) -> list[dict[str, Any]]:
    """Build CLI parameters and input-format guidance from the upstream workflow."""
    nodes_by_id = {node.id: node for node in exist_nodes}
    inputs: list[dict[str, Any]] = []

    for edge in input_edges:
        bind_node = nodes_by_id[edge.source_node]
        parameter = bind_node.data.get('parameter')
        if not isinstance(parameter, str) or not parameter:
            msg = f'bind_param node "{bind_node.id}" has no valid parameter'
            raise ValueError(msg)

        upstream_edge = next((item for item in exist_edges if item.target_node == bind_node.id), None)
        if upstream_edge is None:
            msg = f'bind_param node "{bind_node.id}" is not connected to an upstream file'
            raise ValueError(msg)
        upstream_node = nodes_by_id.get(upstream_edge.source_node)
        if upstream_node is None:
            msg = f'upstream node "{upstream_edge.source_node}" does not exist'
            raise ValueError(msg)
        source_handle = next(
            (handle for handle in upstream_node.output_handles if handle.name == upstream_edge.source_handle),
            None,
        )
        if source_handle is None:
            msg = f'upstream node "{upstream_node.id}" has no output handle named "{upstream_edge.source_handle}"'
            raise ValueError(msg)

        code_input: dict[str, Any] = {
            'parameter': parameter,
            'source_node': upstream_node.name,
            'source_handle': source_handle.name,
            'description': source_handle.description,
        }
        if (
            upstream_node.type == 'subgraph'
            and source_handle.name == 'results_folder'
            and any(
                item.target_node == upstream_node.id
                and item.source_handle == 'collection'
                and (source := nodes_by_id.get(item.source_node)) is not None
                and source.type == 'collect_file_collection'
                for item in exist_edges
            )
        ):
            code_input['format_contract'] = foreach_manifest_contract(
                WorkflowDefinition.model_validate(upstream_node.data['workflow'])
            )
        inputs.append(code_input)

    parameters = [item['parameter'] for item in inputs]
    if len(parameters) != len(set(parameters)):
        msg = f'Code node input parameters must be unique: {parameters}'
        raise ValueError(msg)
    return inputs


def baseline_hash(node_type: CodeNodeType, source: str, dependencies: list[str]) -> str:
    """Fingerprint the exact source and dependency order reviewed by the editor."""
    payload = (
        node_type.value + '\0' + source + '\0' + json.dumps(dependencies, ensure_ascii=False, separators=(',', ':'))
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def read_candidate(
    workspace: Path, node_type: CodeNodeType, *, allow_empty: bool = False
) -> tuple[str, list[str], list[str]]:
    """Read the script and dependencies used to build the CODE proposal."""
    filename = SCRIPT_FILENAMES[node_type]
    source = (workspace / filename).read_text(encoding='utf-8')
    if not source.strip() and not allow_empty:
        message = 'Agent produced an empty script'
        raise ValueError(message)
    dependencies: list[str] = []
    if node_type is CodeNodeType.PYTHON:
        with (workspace / 'pyproject.toml').open('rb') as project_file:
            dependencies = normalize_python_dependencies(tomllib.load(project_file)['project']['dependencies'])
    return source, dependencies, []


def proposal_diff(
    node_type: CodeNodeType,
    before_source: str,
    after_source: str,
    before_dependencies: list[str],
    after_dependencies: list[str],
) -> str:
    """Render one unified diff for the whole turn."""
    filename = SCRIPT_FILENAMES[node_type]
    chunks = list(
        difflib.unified_diff(
            before_source.splitlines(keepends=True),
            after_source.splitlines(keepends=True),
            fromfile=f'a/{filename}',
            tofile=f'b/{filename}',
        )
    )
    if node_type is CodeNodeType.PYTHON:
        dependency_filename = 'pyproject.toml'
        before_lines = tomli_w.dumps({'project': {'dependencies': before_dependencies}}).splitlines(keepends=True)
        after_lines = tomli_w.dumps({'project': {'dependencies': after_dependencies}}).splitlines(keepends=True)
        chunks.extend(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=f'a/{dependency_filename}',
                tofile=f'b/{dependency_filename}',
            )
        )
    return ''.join(chunk if chunk.endswith('\n') else f'{chunk}\n\\ No newline at end of file\n' for chunk in chunks)


async def run_coding_turn(
    request: str,
    *,
    node_type: CodeNodeType,
    user_id: int,
    source: str = '',
    dependencies: list[str] | None = None,
    language: str | None = None,
    history: list[BaseMessage] | None = None,
    turn_id: str | None = None,
    session_id: str | None = None,
    on_progress: ProgressCallback | None = None,
    agent: CodingAgent | None = None,
    input_contract: list[dict[str, Any]] | None = None,
    execution_log: AgentRunLogger | None = None,
) -> CodingTurnResult:
    """Rebuild files, stream native events, validate the candidate, and clean up on cancellation."""
    turn_id = turn_id or str(uuid.uuid4())
    run_log = execution_log or AgentRunLogger(
        'simple-coding', turn_id, session_id=session_id, thread_id=session_id or turn_id
    )
    observation = run_log.observe() if execution_log is None else nullcontext()
    with observation:
        if not request.strip():
            msg = 'A coding request is required.'
            raise ValueError(msg)
        baseline_dependencies = list(dependencies or [])
        with TemporaryDirectory(prefix='bioflow-code-') as directory:
            workspace = Path(directory)
            await initialize_code_workspace_async(workspace, node_type, baseline_dependencies, source=source)
            initial_source, initial_dependencies, _ = read_candidate(workspace, node_type, allow_empty=True)
            context = CodingContext(user_id=user_id, language=language, workspace_path=workspace, node_type=node_type)
            repository_instructions = ''
            if node_type is CodeNodeType.R:
                repository_instructions = (
                    f'CRAN repository for this turn: {get_settings().code.CRAN_REPOSITORY}\n'
                    'If script.R needs additional CRAN packages, check availability and install missing packages '
                    'inside script.R. Use this repository explicitly as the repos argument to install.packages().\n\n'
                )
            prompt = (
                f'Current target CODE node type: {node_type.value}\n'
                f'Authoring mode: {"edit existing source" if source.strip() else "create new script"}\n'
                'The workspace was initialized from the source and dependencies supplied for this turn.\n\n'
                f'{repository_instructions}'
                f'Task request:\n{request}'
            )
            if input_contract is not None:
                prompt += '\n\n' + format_code_inputs(input_contract)
            messages = [*(history or []), HumanMessage(prompt)]
            if agent is None:
                from bioflow_ai.llm.agents.fs_agent.simple_coding import build_simple_coding_agent

                agent = cast('CodingAgent', build_simple_coding_agent())
            final: dict[str, Any] | None = None
            streamed_text = False
            if on_progress is None:
                # Run as a subgraph of the calling agent, whose stream carries our progress and usage; native
                # events would drop them.
                final = await agent.ainvoke({'messages': messages}, config={'recursion_limit': 100}, context=context)
            else:
                agent_run_id = uuid.uuid4()
                async for event in agent.astream_events(
                    {'messages': messages},
                    config={
                        'run_id': agent_run_id,
                        'configurable': {'thread_id': session_id or turn_id},
                        'recursion_limit': 100,
                    },
                    context=context,
                    version='v2',
                ):
                    run_log.native_event(event)
                    kind = event['event']
                    data = event.get('data', {})
                    if kind == 'on_chain_end' and event['run_id'] == str(agent_run_id):
                        final = data['output']
                    elif kind == 'on_chat_model_stream':
                        text = message_text(data.get('chunk'))
                        if text and on_progress is not None:
                            streamed_text = True
                            await on_progress('message.delta', {'text': text})
                    elif kind in {'on_tool_start', 'on_tool_end', 'on_tool_error'} and on_progress is not None:
                        arguments = data.get('input') or {}
                        detail = ''
                        if isinstance(arguments, dict):
                            detail = str(arguments.get('command') or arguments.get('skill_name') or '')
                        await on_progress(
                            'tool.updated',
                            {
                                'call_id': str(event['run_id']),
                                'name': event['name'],
                                'status': {
                                    'on_tool_start': 'running',
                                    'on_tool_end': 'completed',
                                    'on_tool_error': 'failed',
                                }[kind],
                                'detail': detail[:16_000],
                                'output': message_text(data.get('output', data.get('error', '')))[:16_000],
                            },
                        )
            if final is None or final.get('__interrupt__'):
                msg = 'The coding agent did not return a complete conversation.'
                raise ValueError(msg)
            result_messages = final['messages']
            if not streamed_text and on_progress is not None:
                text = '\n\n'.join(
                    message_text(message)
                    for message in result_messages[len(messages) :]
                    if message.type == 'ai' and message_text(message)
                )
                if text:
                    await on_progress('message.delta', {'text': text})
            candidate_source, candidate_dependencies, warnings = read_candidate(
                workspace,
                node_type,
                allow_empty=not source.strip(),
            )
            diff = proposal_diff(node_type, source, candidate_source, baseline_dependencies, candidate_dependencies)
            proposal = None
            changed = (candidate_source, candidate_dependencies) != (initial_source, initial_dependencies)
            if candidate_source.strip() and diff and changed:
                proposal = CodeAgentProposal(
                    id=str(uuid.uuid4()),
                    turn_id=turn_id,
                    baseline_hash=baseline_hash(node_type, source, baseline_dependencies),
                    source=candidate_source,
                    dependencies=candidate_dependencies,
                    diff=diff,
                    warnings=warnings,
                )
            return CodingTurnResult(messages=result_messages, proposal=proposal)
