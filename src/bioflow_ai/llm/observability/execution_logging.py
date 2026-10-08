"""Readable execution records shared by graph, CODE and terminal callers."""

import asyncio
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pprint import pformat
from typing import Any

from loguru import logger

from bioflow_ai.llm.observability.stream_display import select_stream_display

LOG_CONTENT_LIMIT = 2_000
TERMINAL_EVENTS = {'completed', 'waiting_input', 'failed', 'cancelled', 'timed_out'}


@contextmanager
def agent_logging_to_console(*, enabled: bool) -> Iterator[None]:
    """Select console visibility for new execution records in this invocation only."""
    with logger.contextualize(agent_log_console=enabled):
        yield


def agent_record_visible(record: Mapping[str, Any]) -> bool:
    """Keep ordinary logs visible while a terminal renderer owns agent presentation."""
    extra = record['extra']
    return not extra.get('agent_event') or extra.get('agent_log_console', True)


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _preview(value: Any, limit: int = LOG_CONTENT_LIMIT) -> str:
    try:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) <= limit:
        return text
    return f'{text[:limit]}… [truncated; original length={len(text)} chars]'


class AgentRunLogger:
    """Record one invocation without changing graph state or presentation callbacks."""

    def __init__(
        self,
        agent_name: str,
        run_id: str,
        *,
        session_id: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.run_id = run_id
        self.session_id = session_id
        self.thread_id = thread_id
        self._finished = False
        self._logger = logger.bind(
            agent_event=True,
            agent_name=agent_name,
            agent_run_id=run_id,
            agent_session_id=session_id,
            agent_thread_id=thread_id,
        )

    def _record(
        self,
        event: str,
        detail: str = '',
        *,
        namespace: tuple[str, ...] = (),
        level: str = 'INFO',
        error: BaseException | None = None,
    ) -> None:
        path = '/'.join(namespace) or 'root'
        identity = f'agent={self.agent_name} run={self.run_id}'
        if self.session_id:
            identity += f' session={self.session_id}'
        if self.thread_id:
            identity += f' thread={self.thread_id}'
        self._logger.bind(agent_event_type=event, agent_namespace=namespace).opt(exception=error).log(
            level, '[{} ns={} event={}] {}', identity, path, event, detail
        )

    def status(self, event: str, detail: Any = '', *, error: BaseException | None = None) -> None:
        """Record lifecycle outcomes after the caller has committed the actual state."""
        if event in TERMINAL_EVENTS:
            self._finished = True
        level = 'INFO'
        if event in {'failed', 'execution_error'}:
            level = 'ERROR'
        elif event in {'cancelled', 'timed_out'}:
            level = 'WARNING'
        self._record(event, _preview(detail), level=level, error=error)

    @contextmanager
    def observe(self, *, resumed: bool = False) -> Iterator[None]:
        """Observe lifecycle for invocations without an external durable state owner."""
        self._finished = False
        self.status('resumed' if resumed else 'started')
        try:
            yield
        except (asyncio.CancelledError, KeyboardInterrupt):
            self.status('cancelled')
            raise
        except TimeoutError as error:
            self.status('timed_out', str(error), error=error)
            raise
        except Exception as error:
            self.status('failed', str(error), error=error)
            raise
        else:
            if not self._finished:
                self.status('completed')

    def _message(
        self,
        message: Any,
        namespace: tuple[str, ...],
        *,
        call_id: str | None = None,
        updated_keys: tuple[str, ...] = (),
    ) -> None:
        role = _field(message, 'type', _field(message, 'role'))
        # Some streaming providers return an aggregated chunk at model end.
        role = {'AIMessageChunk': 'ai', 'ToolMessageChunk': 'tool'}.get(role, role)
        if role not in {'ai', 'assistant', 'tool'}:
            return
        content = _field(message, 'content', '')
        if role == 'tool':
            failed = _field(message, 'status') == 'error'
            tool_call_id = str(_field(message, 'tool_call_id', call_id))
            name = _field(message, 'name')
            self._record(
                'tool.failed' if failed else 'tool.result',
                f'name={name} call_id={tool_call_id} output={_preview(content)}\n'
                f'Updated keys: {", ".join(updated_keys) if updated_keys else "(none)"}',
                namespace=namespace,
                level='ERROR' if failed else 'INFO',
            )
            return
        metadata = _field(message, 'response_metadata', {}) or {}
        extra = _field(message, 'additional_kwargs', {}) or {}
        reasoning = extra.get('reasoning_content')
        tool_calls = _field(message, 'tool_calls', []) or []
        if content or reasoning or tool_calls:
            detail = f'model={metadata.get("model_name", "")}'
            if call_id:
                detail += f' call_id={call_id}'
            if reasoning:
                detail += f' reasoning={_preview(reasoning)}'
            detail += f' content={content}'
            self._record('model.response', detail, namespace=namespace)
        for tool_call in tool_calls:
            name = _field(tool_call, 'name', 'unknown')
            arguments = _preview(_field(tool_call, 'args', {}))
            tool_call_id = str(_field(tool_call, 'id'))
            self._record(
                'tool.call',
                f'name={name} call_id={tool_call_id} args={arguments}',
                namespace=namespace,
            )

    def stream_chunk(self, chunk: Mapping[str, Any]) -> None:
        """Record the TUI's selected objects with independent, more detailed log previews."""
        display = select_stream_display(chunk)
        if display is None:
            return
        if display.model_message is not None:
            self._message(display.model_message, display.namespace)
        if display.tools_update:
            self._message(
                display.tools_update['messages'][0],
                display.namespace,
                updated_keys=tuple(key for key in display.tools_update if key != 'messages'),
            )
        for name, value in display.middleware:
            self._record('middleware.updated', f'name={name}\n{pformat(value)}', namespace=display.namespace)

    def native_event(self, event: Mapping[str, Any]) -> None:
        """Observe CODE model ends and tool events; token deltas stay with the UI."""
        kind = event.get('event')
        if kind not in {'on_chat_model_end', 'on_tool_start', 'on_tool_end', 'on_tool_error'}:
            return
        data = event.get('data', {})
        metadata = event.get('metadata', {})
        path = metadata.get('langgraph_checkpoint_ns', '')
        namespace = tuple(str(path).split('|')) if path else ()
        call_id = str(event.get('run_id', 'unknown'))
        if kind == 'on_chat_model_end':
            output = data.get('output')
            generations = _field(output, 'generations')
            if generations is not None:
                for batch in generations:
                    for generation in batch if isinstance(batch, (list, tuple)) else [batch]:
                        self._message(_field(generation, 'message', generation), namespace, call_id=call_id)
            else:
                self._message(output, namespace, call_id=call_id)
            return
        name = event.get('name', 'unknown')
        detail = f'name={name} call_id={call_id}'
        if kind == 'on_tool_start':
            self._record('tool.started', f'{detail} args={_preview(data.get("input", {}))}', namespace=namespace)
        else:
            output = data.get('output', data.get('error', ''))
            content = _field(output, 'content', output)
            failed = kind == 'on_tool_error' or _field(output, 'status') == 'error'
            error = data.get('error')
            self._record(
                'tool.failed' if failed else 'tool.result',
                f'{detail} output={_preview(content)}',
                namespace=namespace,
                level='ERROR' if failed else 'INFO',
                error=error if isinstance(error, BaseException) else None,
            )
