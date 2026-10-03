import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ExtendedModelResponse, ModelCallResult, ToolCallRequest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, message_to_dict
from langchain_core.tools import BaseTool
from langgraph.config import get_config
from langgraph.graph import MessagesState
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, SecretStr

from bioflow_ai.llm.observability.trajectories import LocalEventStore, TrajectoryStore

_SENSITIVE_KEYS = {
    'access_key',
    'api_key',
    'authorization',
    'cookie',
    'password',
    'refresh_token',
    'secret',
    'secret_key',
    'token',
}


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).lower().replace('-', '_')
    return normalized in _SENSITIVE_KEYS or normalized.endswith(('_api_key', '_password', '_secret', '_token'))


def _jsonable(value: Any) -> Any:
    """Convert agent values to JSON-compatible objects without serializing credentials."""
    if isinstance(value, SecretStr):
        return '[REDACTED]'
    if isinstance(value, BaseMessage):
        return _jsonable(message_to_dict(value))
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode='json'))
    if isinstance(value, Mapping):
        return {str(key): '[REDACTED]' if _is_sensitive_key(key) else _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (Decimal, Path, uuid.UUID)):
        return str(value)
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, bytes):
        return {'encoding': 'bytes', 'size': len(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _state_without_messages(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if key != 'messages'}


def _tool_schema(tool: BaseTool | dict[str, Any]) -> dict[str, Any]:
    if isinstance(tool, dict):
        return _jsonable(tool)
    try:
        schema = tool.get_input_schema()
        args_schema = schema.model_json_schema()  # pyrefly: ignore [missing-attribute]
    except Exception:
        args_schema = None
    return {
        'name': tool.name,
        'description': tool.description,
        'args_schema': args_schema,
    }


def _model_info(request: ModelRequest) -> dict[str, Any]:
    model = request.model
    model_name = getattr(model, 'model_name', None) or getattr(model, 'model', None)
    return {
        'class': f'{model.__class__.__module__}.{model.__class__.__name__}',
        'name': model_name,
        'metadata': getattr(model, 'metadata', None),
    }


def _model_response_payload(response: ModelCallResult) -> dict[str, Any]:
    if isinstance(response, AIMessage):
        return {'messages': [response], 'structured_response': None}
    if isinstance(response, ExtendedModelResponse):
        return {
            'messages': response.model_response.result,
            'structured_response': response.model_response.structured_response,
            'command': response.command,
        }
    return {
        'messages': response.result,
        'structured_response': response.structured_response,
    }


class SaveMessagesMiddleware(AgentMiddleware):
    """Persist complete model and tool trajectories without affecting agent execution.

    Every event is committed independently through a ``TrajectoryStore``. The default local
    store uses immutable JSON files and atomic renames, so multiple worker processes never
    need to write the same database file.
    """

    def __init__(
        self,
        agent_name: str,
        *,
        store: TrajectoryStore | None = None,
        root_path: str | Path | None = None,
    ):
        self.agent_name = agent_name
        self.store = store or LocalEventStore(root_path)

    @staticmethod
    def _enabled(runtime: Any) -> bool:
        return bool(getattr(runtime.context, 'save_message', True))

    def _identity(self, runtime: Any) -> tuple[str, str | None]:
        try:
            thread_id = get_config().get('configurable', {}).get('thread_id')
        except RuntimeError:
            thread_id = None
        context_id = getattr(runtime.context, 'test_run_id', None)
        # Nested agents inherit the parent LangGraph thread unless explicitly configured
        # otherwise. A test_run_id is the domain-level identity of one tool trial and must
        # therefore take precedence, or multiple tool creations in one workflow are merged.
        identity = str(context_id or thread_id or f'user-{runtime.context.user_id}')
        return f'{self.agent_name}:{identity}', str(thread_id) if thread_id else None

    def _write_event(
        self,
        runtime: Any,
        event_type: str,
        payload: Any,
        *,
        started_at: datetime | None = None,
        duration_ms: float | None = None,
        error: BaseException | None = None,
    ) -> None:
        if not self._enabled(runtime):
            return
        try:
            run_id, thread_id = self._identity(runtime)
            event = _jsonable(
                {
                    'agent': self.agent_name,
                    'run_id': run_id,
                    'thread_id': thread_id,
                    'user_id': getattr(runtime.context, 'user_id', None),
                    'event_type': event_type,
                    'started_at': started_at or datetime.now(UTC),
                    'ended_at': datetime.now(UTC),
                    'duration_ms': duration_ms,
                    'payload': payload,
                    'error': {'type': type(error).__name__, 'message': str(error)} if error else None,
                }
            )
            self.store.append_event(self.agent_name, run_id, event)
        except Exception:
            logger.exception(f'Failed to save a {self.agent_name} trajectory event; agent execution will continue.')

    def before_agent(self, state: MessagesState, runtime: Any) -> dict[str, Any] | None:  # noqa: D102
        self._write_event(
            runtime,
            'agent_start',
            {
                'context': runtime.context,
                'messages': state.get('messages', []),
                'state': _state_without_messages(state),
            },
        )

    async def abefore_agent(self, state: MessagesState, runtime: Any) -> dict[str, Any] | None:  # noqa: D102
        return self.before_agent(state, runtime)

    @staticmethod
    def _model_request_payload(request: ModelRequest) -> dict[str, Any]:
        return {
            'model': _model_info(request),
            'system_message': request.system_message,
            'messages': request.messages,
            'tools': [_tool_schema(tool) for tool in request.tools],
            'tool_choice': request.tool_choice,
            'response_format': request.response_format,
            'model_settings': request.model_settings,
            'state': _state_without_messages(request.state),
        }

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Record one synchronous model request and its response or exception."""
        started_at = datetime.now(UTC)
        start = time.perf_counter()
        request_payload = self._model_request_payload(request)
        try:
            response = handler(request)
        except Exception as error:
            self._write_event(
                request.runtime,
                'model_call',
                {'request': request_payload},
                started_at=started_at,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=error,
            )
            raise
        self._write_event(
            request.runtime,
            'model_call',
            {'request': request_payload, 'response': _model_response_payload(response)},
            started_at=started_at,
            duration_ms=(time.perf_counter() - start) * 1000,
        )
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Record one asynchronous model request and its response or exception."""
        started_at = datetime.now(UTC)
        start = time.perf_counter()
        request_payload = self._model_request_payload(request)
        try:
            response = await handler(request)
        except Exception as error:
            self._write_event(
                request.runtime,
                'model_call',
                {'request': request_payload},
                started_at=started_at,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=error,
            )
            raise
        self._write_event(
            request.runtime,
            'model_call',
            {'request': request_payload, 'response': _model_response_payload(response)},
            started_at=started_at,
            duration_ms=(time.perf_counter() - start) * 1000,
        )
        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Record one synchronous tool request and its observation or exception."""
        started_at = datetime.now(UTC)
        start = time.perf_counter()
        payload: dict[str, Any] = {
            'request': request.tool_call,
            'state': _state_without_messages(request.state),
        }
        try:
            result = handler(request)
        except Exception as error:
            self._write_event(
                request.runtime,
                'tool_call',
                payload,
                started_at=started_at,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=error,
            )
            raise
        payload['response'] = result
        self._write_event(
            request.runtime,
            'tool_call',
            payload,
            started_at=started_at,
            duration_ms=(time.perf_counter() - start) * 1000,
        )
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Record one asynchronous tool request and its observation or exception."""
        started_at = datetime.now(UTC)
        start = time.perf_counter()
        payload: dict[str, Any] = {
            'request': request.tool_call,
            'state': _state_without_messages(request.state),
        }
        try:
            result = await handler(request)
        except Exception as error:
            self._write_event(
                request.runtime,
                'tool_call',
                payload,
                started_at=started_at,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=error,
            )
            raise
        payload['response'] = result
        self._write_event(
            request.runtime,
            'tool_call',
            payload,
            started_at=started_at,
            duration_ms=(time.perf_counter() - start) * 1000,
        )
        return result

    def after_agent(self, state: MessagesState, runtime: Any) -> dict[str, Any] | None:  # noqa: D102
        self._write_event(
            runtime,
            'agent_end',
            {
                'messages': state.get('messages', []),
                'state': _state_without_messages(state),
            },
        )
        logger.debug(f'Saved {self.agent_name} training trajectory.')

    async def aafter_agent(self, state: MessagesState, runtime: Any) -> dict[str, Any] | None:  # noqa: D102
        return self.after_agent(state, runtime)
