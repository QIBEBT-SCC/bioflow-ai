from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from loguru import logger


def _event(kind: str, **details: str | int) -> dict[str, Any]:
    return {kind: details}


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
        return _event('sample_list_project_samples', **common)
    if name == 'set_base_dir':
        base_path = _text_arg(args, 'base_path')
        return _event('sample_set_base_dir', **common, **({'path': base_path} if base_path else {}))
    if name == 'run_command':
        command = _text_arg(args, 'command')
        return _event('sample_run_command', **common, **({'command': command} if command else {}))
    if name == 'add_samples':
        names = args.get('names')
        return _event(
            'sample_add_samples',
            **common,
            **({'count': len(names)} if isinstance(names, list) else {}),
        )
    if name == 'add_sample_file':
        details = {
            key: value
            for key, value in {
                'sample': _text_arg(args, 'sample_name'),
                'tag': _text_arg(args, 'tag'),
            }.items()
            if value is not None
        }
        return _event('sample_add_sample_file', **common, **details)
    if name == 'add_project_file':
        keyword = _text_arg(args, 'keyword')
        return _event(
            'sample_add_project_file',
            **common,
            **({'keyword': keyword} if keyword else {}),
        )
    if name == 'write_conclusion':
        return _event('sample_write_conclusion', **common)
    if name == 'ask_human':
        return _event('sample_ask_human', **common)
    return None


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
