from collections.abc import Awaitable, Callable, Mapping
from pathlib import PurePosixPath
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from loguru import logger


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


def project_planner_progress(request: ToolCallRequest, status: str) -> dict[str, Any] | None:
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


def workflow_builder_progress(request: ToolCallRequest, status: str) -> dict[str, Any] | None:
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


class ProjectPlannerProgressMiddleware(AgentMiddleware):
    """Publish one frontend activity for each project-planner tool call."""

    @staticmethod
    def _emit(request: ToolCallRequest, status: str) -> None:
        payload = project_planner_progress(request, status)
        if payload is None:
            return
        try:
            request.runtime.stream_writer(payload)
        except Exception:
            logger.warning('Unable to publish project-planner progress; agent execution will continue.')

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Publish activity before a synchronous tool call."""
        self._emit(request, 'running')
        result = handler(request)
        self._emit(request, 'completed')
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Publish activity before an asynchronous tool call."""
        self._emit(request, 'running')
        result = await handler(request)
        self._emit(request, 'completed')
        return result


class WorkflowBuilderProgressMiddleware(AgentMiddleware):
    """Publish frontend activities for workflow-builder manager actions."""

    @staticmethod
    def _emit(request: ToolCallRequest, status: str) -> None:
        payload = workflow_builder_progress(request, status)
        if payload is None:
            return
        try:
            request.runtime.stream_writer(payload)
        except Exception:
            logger.warning('Unable to publish Workflow Builder progress; agent execution will continue.')

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Publish activity before and after a synchronous manager call."""
        self._emit(request, 'running')
        result = handler(request)
        self._emit(request, 'completed')
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Publish activity before and after an asynchronous manager call."""
        self._emit(request, 'running')
        result = await handler(request)
        self._emit(request, 'completed')
        return result
