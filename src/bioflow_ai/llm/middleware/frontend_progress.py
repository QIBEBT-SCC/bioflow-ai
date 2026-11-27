import functools
import uuid
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from pathlib import PurePosixPath
from typing import Any, Literal

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config, get_stream_writer
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command
from loguru import logger

ProgressStatus = Literal['running', 'completed', 'error']
ProgressTranslator = Callable[[ToolCallRequest, ProgressStatus], dict[str, Any] | None]

TASK_NAMESPACE_KEY = 'task_ns'


def _task_namespace(config: RunnableConfig) -> str | None:
    """Return the checkpoint namespace of the running task, which prefixes the namespace of graphs it invokes."""
    namespace = config.get('configurable', {}).get('checkpoint_ns')
    return namespace if isinstance(namespace, str) and namespace else None


def _with_task_namespace(payload: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """Attach the task namespace so consumers can nest the activities of sub-agents under this step."""
    namespace = _task_namespace(config)
    return {**payload, TASK_NAMESPACE_KEY: namespace} if namespace else payload


def _args(request: ToolCallRequest) -> Mapping[str, Any]:
    args = request.tool_call.get('args', {})
    return args if isinstance(args, Mapping) else {}


def _text_arg(args: Mapping[str, Any], name: str) -> str | None:
    value = args.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


PATCH_FILE_HEADERS = ('*** Add File: ', '*** Update File: ', '*** Delete File: ')


def patch_files(patch: str | None) -> str | None:
    """Return the deduplicated file names an ``apply_patch`` patch adds, updates, or deletes."""
    if not patch:
        return None
    names = [
        PurePosixPath(line.split(': ', 1)[1].strip()).name
        for line in patch.splitlines()
        if line.startswith(PATCH_FILE_HEADERS)
    ]
    return ', '.join(dict.fromkeys(name for name in names if name)) or None


def _web_query(args: Mapping[str, Any]) -> str | None:
    queries = args.get('search_query')
    if isinstance(queries, list) and queries and isinstance(queries[0], Mapping):
        return _text_arg(queries[0], 'q')
    return None


def _planner_details(name: str, args: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the activity details for one project-planner tool, or ``None`` when it is not displayed."""
    if name in {'list_project_samples', 'view_pdf', 'confirm_plan', 'ask_human'}:
        return {}
    if name == 'set_base_dir':
        base_path = _text_arg(args, 'base_path')
        return {'path': base_path} if base_path else {}
    if name == 'run_command':
        command = _text_arg(args, 'command')
        return {'command': command} if command else {}
    if name == 'add_samples':
        names = args.get('names')
        return {'count': len(names)} if isinstance(names, list) else {}
    if name == 'add_sample_file':
        return {
            key: value
            for key, value in {'sample': _text_arg(args, 'sample_name'), 'tag': _text_arg(args, 'tag')}.items()
            if value is not None
        }
    if name == 'add_project_file':
        keyword = _text_arg(args, 'keyword')
        return {'keyword': keyword} if keyword else {}
    if name == 'apply_patch':
        files = patch_files(_text_arg(args, 'patch'))
        return {'path': files} if files else {}
    if name == 'web_run':
        query = _web_query(args)
        return {'subject': query} if query else {}
    return None


PLANNER_ACTIVITY_NAMES = {'apply_patch': 'write_file'}


def project_planner_progress(request: ToolCallRequest, status: ProgressStatus) -> dict[str, Any] | None:
    """Translate one project-planner tool call into a ``planner_*`` frontend activity."""
    name = request.tool_call.get('name')
    details = _planner_details(name, _args(request)) if isinstance(name, str) else None
    if details is None:
        return None
    activity = PLANNER_ACTIVITY_NAMES.get(name, name)
    return {f'planner_{activity}': {'tool_call_id': request.tool_call['id'], 'status': status, **details}}


WORKFLOW_MANAGERS = {
    'tool_node_manager': 'tool_node',
    'resource_node_manager': 'resource_node',
    'utility_node_manager': 'utility_node',
    'workflow_manager': 'workflow',
    'subgraph_node_manager': 'subgraph',
    'subgraph_interface_manager': 'subgraph',
}

WORKFLOW_OPERATIONS = {
    'search': 'search',
    'search_db': 'search',
    'view': 'inspect',
    'view_marks': 'inspect',
    'view_workflow': 'inspect',
    'view_node': 'inspect',
    'read_node_memory': 'inspect',
    'list_workflow_drafts': 'list',
    'add': 'add',
    'add_processor': 'add',
    'add_exist_code': 'add',
    'add_new_code': 'create',
    'add_note': 'add',
    'add_downstream_summary': 'add',
    'create': 'create',
    'create_db': 'create',
    'create_workflow_draft': 'create',
    'update_args': 'update',
    'update_workflow_metadata': 'update',
    'switch_workflow_draft': 'switch',
    'delete_workflow_draft': 'delete',
    'delete_node': 'delete',
    'fix_edge': 'fix',
    'save_all_workflows': 'save',
    'add_subgraph_inputs': 'add',
    'remove_subgraph_inputs': 'delete',
    'set_subgraph_outputs': 'update',
    'save_subgraph': 'save',
}

SUBJECT_FIELDS = (
    'name',
    'query',
    'resource_type',
    'processor_name',
    'code_uid',
    'node_id',
    'draft_id',
    'selected_id',
    'uid',
)


SEARCH_RESULT_COUNT_KEY = 'result_count'
MEMORY_DESCRIPTION_KEY = 'memory_description'


def _search_keywords(params: Mapping[str, Any]) -> list[str]:
    """Return the keywords of a manager search: tool-node command patterns or a free-text query."""
    patterns = params.get('patterns')
    if isinstance(patterns, list):
        return [pattern.strip() for pattern in patterns if isinstance(pattern, str) and pattern.strip()]
    query = params.get('query')
    return [query.strip()] if isinstance(query, str) and query.strip() else []


def _subject(params: Mapping[str, Any]) -> str | None:
    for field in SUBJECT_FIELDS:
        value = params.get(field)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()
    return None


def workflow_builder_progress(request: ToolCallRequest, status: ProgressStatus) -> dict[str, Any] | None:
    """Translate one workflow-builder manager call into a frontend activity."""
    tool_name = request.tool_call.get('name')
    manager = WORKFLOW_MANAGERS.get(tool_name)
    if manager is None:
        return None

    args = _args(request)
    action = _text_arg(args, 'action')
    if action is None or action == 'help':
        return None
    params_value = args.get('params', {})
    params = params_value if isinstance(params_value, Mapping) else {}
    subject = _subject(params)
    operation = WORKFLOW_OPERATIONS.get(action, 'update')
    keywords = _search_keywords(params) if operation == 'search' else []
    return {
        'workflow_manager_action': {
            'tool_call_id': request.tool_call['id'],
            'status': status,
            'manager': manager,
            'action': action,
            'operation': operation,
            **({'subject': subject} if subject else {}),
            **({'keywords': keywords} if keywords else {}),
        }
    }


def _agent_tool_details(name: str, args: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the activity details for one generic agent tool, or ``None`` when it is not displayed."""
    if name in {
        'ask_human',
        'summarize_work',
        'write_diagnosis',
        'read_node_memory',
        'write_node_memory',
        'finalize_workflow_fix',
    }:
        return {}
    if name in {'workflow_reader', 'fix_subgraph'}:
        details = {'action': _text_arg(args, 'action'), 'subject': _text_arg(args, 'node_id')}
        return {key: value for key, value in details.items() if value is not None}
    if name == 'project_workflow_reader':
        action = _text_arg(args, 'action')
        subject = _text_arg(args, 'node_id') if action == 'node' else _text_arg(args, 'workflow_uid')
        details = {'action': action, 'subject': subject}
        return {key: value for key, value in details.items() if value is not None}
    if name == 'run_command':
        details = {'subject': _text_arg(args, 'source'), 'command': _text_arg(args, 'command')}
        return {key: value for key, value in details.items() if value is not None}
    if name == 'view_image':
        source = _text_arg(args, 'source')
        path = _text_arg(args, 'path')
        subject = f'{source}/{path}' if source and path and path != '.' else source
        return {'subject': subject} if subject else {}
    return None


TOOL_CREATION_SUBJECT_FIELDS = {
    'search_local_image': 'query',
    'search_biodb': 'query',
    'search_tools': 'query',
    'select_image': 'image_ref',
}
TOOL_CREATION_PLAIN_TOOLS = {
    'ask_human',
    'return_to_parent',
    'write_tool_report',
    'select_tool',
    'select_classification',
    'set_file_mounts',
    'set_command_detail',
    'set_tool_info',
    'search_node',
    'view_node',
    'inspect_files',
    'execute_node',
}
TOOL_CREATION_ACTIVITY_NAMES = {'run_command': 'container_command', 'search_tools': 'search_tools_by_function'}


def _tool_creation_details(name: str, args: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the activity details for one tool-creation sub-agent tool, or ``None`` when it is not displayed."""
    if name in TOOL_CREATION_PLAIN_TOOLS:
        return {}
    if field := TOOL_CREATION_SUBJECT_FIELDS.get(name):
        subject = _text_arg(args, field)
        return {'subject': subject} if subject else {}
    if name == 'read_tool_doc':
        tool_id = args.get('tool_id')
        return {'subject': str(tool_id)} if isinstance(tool_id, int) else {}
    if name == 'run_command':
        details = {'subject': _text_arg(args, 'image_ref'), 'command': _text_arg(args, 'command')}
        return {key: value for key, value in details.items() if value is not None}
    return None


def tool_creation_progress(request: ToolCallRequest, status: ProgressStatus) -> dict[str, Any] | None:
    """Translate one tool call of a tool-creation sub-agent into an ``agent_tool`` frontend activity."""
    name = request.tool_call.get('name')
    details = _tool_creation_details(name, _args(request)) if isinstance(name, str) else None
    if details is None:
        return None
    activity = TOOL_CREATION_ACTIVITY_NAMES.get(name, name)
    return {'agent_tool': {'tool_call_id': request.tool_call['id'], 'status': status, 'tool': activity, **details}}


def agent_tool_progress(request: ToolCallRequest, status: ProgressStatus) -> dict[str, Any] | None:
    """Translate one generic agent tool call into an ``agent_tool`` frontend activity."""
    name = request.tool_call.get('name')
    details = _agent_tool_details(name, _args(request)) if isinstance(name, str) else None
    if details is None:
        return None
    return {'agent_tool': {'tool_call_id': request.tool_call['id'], 'status': status, 'tool': name, **details}}


def coding_progress(request: ToolCallRequest, status: ProgressStatus) -> dict[str, Any] | None:
    """Translate one coding sub-agent tool call into an ``agent_tool`` frontend activity."""
    name = request.tool_call.get('name')
    args = _args(request)
    if name == 'run_command':
        details = {'tool': 'container_command', 'command': _text_arg(args, 'command')}
    elif name == 'apply_patch':
        details = {'tool': 'apply_patch', 'path': patch_files(_text_arg(args, 'patch'))}
    else:
        return None
    details = {key: value for key, value in details.items() if value is not None}
    return {'agent_tool': {'tool_call_id': request.tool_call['id'], 'status': status, **details}}


SKILL_TOOLS = {'load_skill': 'skill_name', 'read_skill_resource': 'resource_path'}


def skill_progress(request: ToolCallRequest, status: ProgressStatus) -> dict[str, Any] | None:
    """Translate one skill tool call into an ``agent_tool`` frontend activity."""
    name = request.tool_call.get('name')
    field = SKILL_TOOLS.get(name) if isinstance(name, str) else None
    if field is None:
        return None
    subject = _text_arg(_args(request), field)
    return {
        'agent_tool': {
            'tool_call_id': request.tool_call['id'],
            'status': status,
            'tool': name,
            **({'subject': subject} if subject else {}),
        }
    }


def _result_status(result: ToolMessage | Command[Any]) -> ProgressStatus:
    """Report ``error`` when the tool result carries an error ``ToolMessage``."""
    if isinstance(result, ToolMessage):
        messages: list[Any] = [result]
    else:
        update = result.update
        messages = list(update.get('messages', [])) if isinstance(update, Mapping) else []
    failed = any(isinstance(message, ToolMessage) and message.status == 'error' for message in messages)
    return 'error' if failed else 'completed'


def _result_details(result: ToolMessage | Command[Any]) -> dict[str, Any]:
    """Collect display details a tool attached to its result message as ``artifact``."""
    if isinstance(result, ToolMessage):
        messages: list[Any] = [result]
    else:
        update = result.update
        messages = list(update.get('messages', [])) if isinstance(update, Mapping) else []
    for message in messages:
        artifact = message.artifact if isinstance(message, ToolMessage) else None
        if isinstance(artifact, Mapping) and isinstance(artifact.get(SEARCH_RESULT_COUNT_KEY), int):
            return {SEARCH_RESULT_COUNT_KEY: artifact[SEARCH_RESULT_COUNT_KEY]}
        if isinstance(artifact, Mapping) and isinstance(artifact.get(MEMORY_DESCRIPTION_KEY), str):
            return {MEMORY_DESCRIPTION_KEY: artifact[MEMORY_DESCRIPTION_KEY]}
    return {}


def _emit_progress(
    request: ToolCallRequest,
    translate: ProgressTranslator,
    status: ProgressStatus,
    result: ToolMessage | Command[Any] | None = None,
) -> None:
    payload = translate(request, status)
    if payload is None:
        return
    details = _result_details(result) if result is not None else {}
    if details:
        payload = {key: {**value, **details} if isinstance(value, dict) else value for key, value in payload.items()}
    try:
        request.runtime.stream_writer(_with_task_namespace(payload, request.runtime.config))
    except Exception:
        logger.warning('Unable to publish agent progress; agent execution will continue.')


def call_with_progress(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    translate: ProgressTranslator,
) -> ToolMessage | Command[Any]:
    """Run a synchronous tool call between its running and finished activities.

    Interrupts propagate without a finished activity, so the resumed call completes the same step.
    """
    _emit_progress(request, translate, 'running')
    try:
        result = handler(request)
    except GraphBubbleUp:
        raise
    except Exception:
        _emit_progress(request, translate, 'error')
        raise
    _emit_progress(request, translate, _result_status(result), result)
    return result


async def acall_with_progress(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    translate: ProgressTranslator,
) -> ToolMessage | Command[Any]:
    """Run an asynchronous tool call between its running and finished activities."""
    _emit_progress(request, translate, 'running')
    try:
        result = await handler(request)
    except GraphBubbleUp:
        raise
    except Exception:
        _emit_progress(request, translate, 'error')
        raise
    _emit_progress(request, translate, _result_status(result), result)
    return result


class ToolProgressMiddleware(AgentMiddleware[AgentState[Any], Any]):
    """Publish running and finished frontend activities for the tool calls ``translate`` displays."""

    @staticmethod
    def translate(request: ToolCallRequest, status: ProgressStatus) -> dict[str, Any] | None:
        """Return the activity payload for one tool call, or ``None`` when it is not displayed."""
        return agent_tool_progress(request, status)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Publish activity around a synchronous tool call."""
        return call_with_progress(request, handler, self.translate)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Publish activity around an asynchronous tool call."""
        return await acall_with_progress(request, handler, self.translate)


class ProjectPlannerProgressMiddleware(ToolProgressMiddleware):
    """Publish one frontend activity for each project-planner tool call."""

    @staticmethod
    def translate(request: ToolCallRequest, status: ProgressStatus) -> dict[str, Any] | None:  # noqa: D102
        return project_planner_progress(request, status)


class ToolCreationProgressMiddleware(ToolProgressMiddleware):
    """Publish frontend activities for the tool calls of the tool-creation sub-agents."""

    @staticmethod
    def translate(request: ToolCallRequest, status: ProgressStatus) -> dict[str, Any] | None:  # noqa: D102
        return tool_creation_progress(request, status)


class CodingProgressMiddleware(ToolProgressMiddleware):
    """Publish frontend activities for the tool calls of the coding sub-agent."""

    @staticmethod
    def translate(request: ToolCallRequest, status: ProgressStatus) -> dict[str, Any] | None:  # noqa: D102
        return coding_progress(request, status)


class WorkflowBuilderProgressMiddleware(ToolProgressMiddleware):
    """Publish frontend activities for workflow manager actions and the other workflow agent tools."""

    @staticmethod
    def translate(request: ToolCallRequest, status: ProgressStatus) -> dict[str, Any] | None:  # noqa: D102
        return workflow_builder_progress(request, status) or agent_tool_progress(request, status)


def tool_generator_step[**P, R](
    step: str,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
    """Publish a ``tool_generator_step`` activity around one asynchronous tool-generator graph node."""

    def decorate(node: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R]]:
        @functools.wraps(node)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            call_id = f'tool-generator:{step}:{uuid.uuid4().hex}'

            def emit(status: ProgressStatus) -> None:
                try:
                    get_stream_writer()(
                        _with_task_namespace(
                            {'tool_generator_step': {'tool_call_id': call_id, 'status': status, 'step': step}},
                            get_config(),
                        )
                    )
                except Exception:
                    logger.warning('Unable to publish tool-generator progress; agent execution will continue.')

            emit('running')
            try:
                result = await node(*args, **kwargs)
            except GraphBubbleUp:
                raise
            except Exception:
                emit('error')
                raise
            emit('completed')
            return result

        return wrapper

    return decorate
