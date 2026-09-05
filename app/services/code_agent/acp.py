"""Small ACP v1 JSON-RPC client for audited code-agent adapters."""

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import PurePosixPath
from typing import Any

ACP_PROTOCOL_VERSION = 1
ACP_WORKSPACE = PurePosixPath('/workspace')


class AcpError(RuntimeError):
    """Base error raised by the ACP transport."""


class AcpProtocolError(AcpError):
    """Raised for malformed or incompatible ACP messages."""


class AcpProcessError(AcpError):
    """Raised when the adapter process exits unexpectedly."""


UpdateHandler = Callable[[dict[str, Any]], Awaitable[None]]
ElicitationHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def _path_is_inside_workspace(value: str) -> bool:
    path = PurePosixPath(value)
    if not path.is_absolute():
        path = ACP_WORKSPACE / path
    normalized = PurePosixPath('/')
    for part in path.parts[1:]:
        if part == '..':
            normalized = normalized.parent
        elif part not in {'', '.'}:
            normalized /= part
    return normalized == ACP_WORKSPACE or ACP_WORKSPACE in normalized.parents


def permission_is_safe(params: dict[str, Any]) -> bool:
    """Approve only requests whose declared paths stay inside the isolated workspace."""
    tool_call = params.get('toolCall')
    if not isinstance(tool_call, dict):
        return False
    locations = tool_call.get('locations') or []
    if locations:
        locations_are_safe = all(
            isinstance(location, dict)
            and isinstance(location.get('path'), str)
            and _path_is_inside_workspace(location['path'])
            for location in locations
        )
        if not locations_are_safe:
            return False
    raw_input = tool_call.get('rawInput')
    if not isinstance(raw_input, dict):
        return bool(locations)
    cwd = raw_input.get('cwd', '/workspace')
    if not isinstance(cwd, str) or not _path_is_inside_workspace(cwd):
        return False
    command = raw_input.get('command')
    command_text = command if isinstance(command, str) else json.dumps(command, ensure_ascii=False)
    denied = (
        'sudo ',
        'su ',
        '--privileged',
        '/var/run/docker.sock',
        '/run/podman/podman.sock',
        '$HOME',
        '${HOME}',
    )
    if any(fragment in command_text for fragment in denied):
        return False
    if re.search(r'(^|[\s"\'=:])(\.\./|~(?:/|\s|$))', command_text):
        return False
    return re.search(r'(^|[\s"\'=:])/(?!workspace(?:/|\b))', command_text) is None


class AcpClient:
    """Multiplex JSON-RPC requests and ACP updates over an adapter's stdio."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        on_update: UpdateHandler,
        on_elicitation: ElicitationHandler | None = None,
    ) -> None:
        self.process = process
        self.on_update = on_update
        self.on_elicitation = on_elicitation
        self._next_id = 1
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task = asyncio.create_task(self._read_loop(), name='acp-stdio-reader')

    async def _write(self, payload: dict[str, Any]) -> None:
        if self.process.stdin is None:
            message = 'ACP stdin is unavailable'
            raise AcpProcessError(message)
        self.process.stdin.write(json.dumps(payload, separators=(',', ':')).encode() + b'\n')
        await self.process.stdin.drain()

    async def _read_loop(self) -> None:
        if self.process.stdout is None:
            return
        try:
            while line := await self.process.stdout.readline():
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as error:
                    error_message = 'ACP adapter emitted malformed JSON'
                    raise AcpProtocolError(error_message) from error
                if not isinstance(message, dict) or message.get('jsonrpc') != '2.0':
                    error_message = 'ACP adapter emitted an invalid JSON-RPC message'
                    raise AcpProtocolError(error_message)  # noqa: TRY301
                message_id = message.get('id')
                if message_id is not None and ('result' in message or 'error' in message):
                    pending = self._pending.pop(message_id, None)
                    if pending is not None and not pending.done():
                        pending.set_result(message)
                    continue
                method = message.get('method')
                params = message.get('params')
                if method == 'session/update' and isinstance(params, dict):
                    await self.on_update(params)
                elif message_id is not None and isinstance(method, str):
                    await self._handle_request(message_id, method, params)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            for pending in self._pending.values():
                if not pending.done():
                    pending.set_exception(error)
            self._pending.clear()
        else:
            error = AcpProcessError(f'ACP adapter exited with status {self.process.returncode}')
            for pending in self._pending.values():
                if not pending.done():
                    pending.set_exception(error)
            self._pending.clear()

    async def _handle_request(self, request_id: int, method: str, params: object) -> None:
        if method == 'elicitation/create' and isinstance(params, dict) and self.on_elicitation is not None:
            result = await self.on_elicitation(params)
            await self._write({'jsonrpc': '2.0', 'id': request_id, 'result': result})
            return
        if method != 'session/request_permission' or not isinstance(params, dict):
            await self._write(
                {
                    'jsonrpc': '2.0',
                    'id': request_id,
                    'error': {'code': -32601, 'message': 'Client method not supported'},
                }
            )
            return
        options = params.get('options')
        selected: str | None = None
        if permission_is_safe(params) and isinstance(options, list):
            for option in options:
                if isinstance(option, dict) and option.get('kind') == 'allow_once':
                    selected = option.get('optionId')
                    break
        result = (
            {'outcome': {'outcome': 'selected', 'optionId': selected}}
            if selected
            else {'outcome': {'outcome': 'cancelled'}}
        )
        await self._write({'jsonrpc': '2.0', 'id': request_id, 'result': result})

    async def request(self, method: str, params: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        """Send one ACP request and return its result."""
        request_id = self._next_id
        self._next_id += 1
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._write({'jsonrpc': '2.0', 'id': request_id, 'method': method, 'params': params})
            response = await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending.pop(request_id, None)
        if 'error' in response:
            error = response['error']
            message = error.get('message', 'Unknown ACP error') if isinstance(error, dict) else str(error)
            raise AcpProtocolError(message)
        result = response.get('result')
        if not isinstance(result, dict):
            message = f'ACP method {method} returned an invalid result'
            raise AcpProtocolError(message)
        return result

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        """Send one ACP notification."""
        await self._write({'jsonrpc': '2.0', 'method': method, 'params': params})

    async def initialize(self, *, timeout: float, enable_url_elicitation: bool = False) -> dict[str, Any]:
        """Negotiate stable ACP v1 and the optional out-of-band URL capability."""
        capabilities: dict[str, Any] = {
            'fs': {'readTextFile': False, 'writeTextFile': False},
            'terminal': False,
        }
        if enable_url_elicitation:
            capabilities['elicitation'] = {'url': {}}
        response = await self.request(
            'initialize',
            {
                'protocolVersion': ACP_PROTOCOL_VERSION,
                'clientCapabilities': capabilities,
                'clientInfo': {'name': 'bioflow-ai', 'title': 'BioFlowAI', 'version': '0.1.0'},
            },
            timeout=timeout,
        )
        if response.get('protocolVersion') != ACP_PROTOCOL_VERSION:
            message = 'ACP adapter does not support protocol v1'
            raise AcpProtocolError(message)
        return response

    async def authenticate(self, method_id: str, *, timeout: float) -> None:
        """Authenticate using one method explicitly selected by BioFlowAI."""
        await self.request('authenticate', {'methodId': method_id}, timeout=timeout)

    async def authentication_status(self, *, timeout: float) -> dict[str, Any]:
        """Read the adapter's secret-free Codex account status extension."""
        return await self.request('authentication/status', {}, timeout=timeout)

    async def new_session(self, *, timeout: float) -> str:
        """Create one ACP session rooted at the isolated workspace."""
        response = await self.request(
            'session/new',
            {'cwd': '/workspace', 'additionalDirectories': [], 'mcpServers': []},
            timeout=timeout,
        )
        session_id = response.get('sessionId')
        if not isinstance(session_id, str):
            message = 'ACP session/new returned no sessionId'
            raise AcpProtocolError(message)
        return session_id

    async def prompt(self, session_id: str, text: str, *, timeout: float) -> dict[str, Any]:
        """Run one text prompt to completion."""
        return await self.request(
            'session/prompt',
            {'sessionId': session_id, 'prompt': [{'type': 'text', 'text': text}]},
            timeout=timeout,
        )

    async def cancel(self, session_id: str) -> None:
        """Cancel the active prompt."""
        await self.notify('session/cancel', {'sessionId': session_id})

    async def close(self) -> None:
        """Stop the transport and adapter process."""
        if self.process.stdin is not None:
            self.process.stdin.close()
            with suppress(BrokenPipeError, ConnectionResetError):
                await self.process.stdin.wait_closed()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=3)
        except TimeoutError:
            with suppress(ProcessLookupError):
                self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=3)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    self.process.kill()
                await self.process.wait()
        self._reader_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._reader_task
