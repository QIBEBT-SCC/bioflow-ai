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


def _tool_progress(request: ToolCallRequest, status: str) -> dict[str, Any] | None:
    name = request.tool_call.get('name')
    args = _args(request)
    common = {'tool_call_id': request.tool_call['id'], 'status': status}

    if name == 'list_project_samples':
        return {'sample_list_project_samples': common}
    if name == 'set_base_dir':
        base_path = _text_arg(args, 'base_path')
        return {
            'sample_set_base_dir': {
                **common,
                **({'path': base_path} if base_path else {}),
            }
        }
    if name == 'run_command':
        command = _text_arg(args, 'command')
        return {
            'sample_run_command': {
                **common,
                **({'command': command} if command else {}),
            }
        }
    if name == 'add_samples':
        names = args.get('names')
        return {
            'sample_add_samples': {
                **common,
                **({'count': len(names)} if isinstance(names, list) else {}),
            }
        }
    if name == 'add_sample_file':
        details = {
            key: value
            for key, value in {
                'sample': _text_arg(args, 'sample_name'),
                'tag': _text_arg(args, 'tag'),
            }.items()
            if value is not None
        }
        return {'sample_add_sample_file': {**common, **details}}
    if name == 'add_project_file':
        keyword = _text_arg(args, 'keyword')
        return {
            'sample_add_project_file': {
                **common,
                **({'keyword': keyword} if keyword else {}),
            }
        }
    if name == 'write_conclusion':
        return {'sample_write_conclusion': common}
    if name == 'ask_human':
        return {'sample_ask_human': common}
    return None


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


def project_prepare_progress(request: ToolCallRequest, status: str) -> dict[str, Any] | None:
    """Translate one project-prepare tool call into a frontend activity.

    Sample registration tools reuse the Sample Manager activities; planning tools get ``prepare_*`` activities.
    """
    name = request.tool_call.get('name')
    args = _args(request)
    common = {'tool_call_id': request.tool_call['id'], 'status': status}

    if name == 'apply_patch':
        files = _patch_files(_text_arg(args, 'patch'))
        return {'prepare_write_file': {**common, **({'path': files} if files else {})}}
    if name == 'web_run':
        query = _web_query(args)
        return {'prepare_web_run': {**common, **({'subject': query} if query else {})}}
    if name == 'view_pdf':
        return {'prepare_view_pdf': common}
    if name == 'confirm_plan':
        return {'prepare_confirm_plan': common}
    return _tool_progress(request, status)


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


class SampleManagerProgressMiddleware(AgentMiddleware):
    """Publish one frontend activity for each Sample Manager tool call."""

    @staticmethod
    def _emit(request: ToolCallRequest, status: str) -> None:
        payload = _tool_progress(request, status)
        if payload is None:
            return
        try:
            request.runtime.stream_writer(payload)
        except Exception:
            logger.warning('Unable to publish Sample Manager progress; agent execution will continue.')

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


class ProjectPrepareProgressMiddleware(SampleManagerProgressMiddleware):
    """Publish one frontend activity for each project-prepare tool call."""

    @staticmethod
    def _emit(request: ToolCallRequest, status: str) -> None:
        payload = project_prepare_progress(request, status)
        if payload is None:
            return
        try:
            request.runtime.stream_writer(payload)
        except Exception:
            logger.warning('Unable to publish project-prepare progress; agent execution will continue.')
