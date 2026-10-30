import functools
import uuid
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from pathlib import PurePosixPath
from typing import Any, Literal

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.config import get_stream_writer
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command
from loguru import logger

ProgressStatus = Literal['running', 'completed', 'error']
ProgressTranslator = Callable[[ToolCallRequest, ProgressStatus], dict[str, Any] | None]


def _args(request: ToolCallRequest) -> Mapping[str, Any]:
    args = request.tool_call.get('args', {})
    return args if isinstance(args, Mapping) else {}


def _text_arg(args: Mapping[str, Any], name: str) -> str | None:
    value = args.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


PATCH_FILE_HEADERS = ('*** Add File: ', '*** Update File: ', '*** Delete File: ')


def _patch_files(patch: str | None) -> str | None:
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
        files = _patch_files(_text_arg(args, 'patch'))
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
    return {
        'workflow_manager_action': {
            'tool_call_id': request.tool_call['id'],
            'status': status,
            'manager': manager,
            'action': action,
            'operation': WORKFLOW_OPERATIONS.get(action, 'update'),
            **({'subject': subject} if subject else {}),
        }
    }


def _agent_tool_details(name: str, args: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the activity details for one generic agent tool, or ``None`` when it is not displayed."""
    if name in {'ask_human', 'summarize_work', 'write_diagnosis', 'finalize_workflow_fix'}:
        return {}
    if name in {'workflow_reader', 'fix_subgraph'}:
        details = {'action': _text_arg(args, 'action'), 'subject': _text_arg(args, 'node_id')}
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


def agent_tool_progress(request: ToolCallRequest, status: ProgressStatus) -> dict[str, Any] | None:
    """Translate one generic agent tool call into an ``agent_tool`` frontend activity."""
    name = request.tool_call.get('name')
    details = _agent_tool_details(name, _args(request)) if isinstance(name, str) else None
    if details is None:
        return None
    return {'agent_tool': {'tool_call_id': request.tool_call['id'], 'status': status, 'tool': name, **details}}


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


def _emit_progress(request: ToolCallRequest, translate: ProgressTranslator, status: ProgressStatus) -> None:
    payload = translate(request, status)
    if payload is None:
        return
    try:
        request.runtime.stream_writer(payload)
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
    _emit_progress(request, translate, _result_status(result))
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
    _emit_progress(request, translate, _result_status(result))
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
                        {'tool_generator_step': {'tool_call_id': call_id, 'status': status, 'step': step}}
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
