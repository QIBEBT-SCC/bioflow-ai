import asyncio
import contextlib
import json
import uuid
from typing import Any

import websockets
from websockets import ClientConnection


class WeComClientError(RuntimeError):
    """Raised when the WeCom bot gateway rejects a request."""


class WeComClient:
    """Minimal raw client for the WeCom intelligent-bot WebSocket protocol."""

    def __init__(self, *, bot_id: str, secret: str, ws_url: str) -> None:
        self._bot_id = bot_id
        self._secret = secret
        self._ws_url = ws_url
        self._socket: ClientConnection | None = None
        self._runner: asyncio.Task[None] | None = None
        self._authenticated = asyncio.Event()
        self._closed = False
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._send_lock = asyncio.Lock()
        self.last_error: str | None = None

    async def start(self) -> None:
        """Start the reconnect loop and wait for authentication."""
        if self._closed:
            msg = 'WeCom client is closed.'
            raise WeComClientError(msg)
        if self._runner is None or self._runner.done():
            self._runner = asyncio.create_task(self._run(), name='wecom-notification-client')
        try:
            await asyncio.wait_for(self._authenticated.wait(), timeout=25)
        except TimeoutError as error:
            detail = self.last_error or 'connection timed out'
            message = f'WeCom authentication failed: {detail}'
            raise WeComClientError(message) from error

    async def send_markdown(self, *, chat_id: str, content: str) -> None:
        """Send one Markdown notification to a group chat."""
        await self.start()
        await self._request(
            'aibot_send_msg',
            {
                'chatid': chat_id,
                'chat_type': 2,
                'msgtype': 'markdown',
                'markdown': {'content': content},
            },
        )

    async def close(self) -> None:
        """Stop reconnecting and release the WebSocket."""
        self._closed = True
        self._authenticated.clear()
        if self._socket is not None:
            await self._socket.close()
        if self._runner is not None:
            self._runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner
        self._fail_pending(WeComClientError('WeCom client closed.'))

    async def _run(self) -> None:
        retry_delay = 1
        while not self._closed:
            receiver: asyncio.Task[None] | None = None
            heartbeat: asyncio.Task[None] | None = None
            try:
                async with websockets.connect(
                    self._ws_url,
                    open_timeout=20,
                    close_timeout=5,
                    ping_interval=None,
                ) as socket:
                    self._socket = socket
                    receiver = asyncio.create_task(self._receive_loop(socket))
                    await self._request(
                        'aibot_subscribe',
                        {'bot_id': self._bot_id, 'secret': self._secret},
                    )
                    self.last_error = None
                    retry_delay = 1
                    self._authenticated.set()
                    heartbeat = asyncio.create_task(self._heartbeat_loop())
                    done, _pending = await asyncio.wait(
                        {receiver, heartbeat},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        await task
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.last_error = f'{type(error).__name__}: {error}'
            finally:
                self._authenticated.clear()
                self._socket = None
                if heartbeat is not None:
                    heartbeat.cancel()
                if receiver is not None:
                    receiver.cancel()
                await asyncio.gather(
                    *(task for task in (heartbeat, receiver) if task is not None),
                    return_exceptions=True,
                )
                self._fail_pending(WeComClientError(self.last_error or 'WeCom connection closed.'))
            if not self._closed:
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)

    async def _receive_loop(self, socket: ClientConnection) -> None:
        async for raw_message in socket:
            message = json.loads(raw_message)
            request_id = message.get('headers', {}).get('req_id')
            if not request_id:
                continue
            future = self._pending.pop(request_id, None)
            if future is not None and not future.done():
                future.set_result(message)

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            await self._request('ping')

    async def _request(self, command: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        socket = self._socket
        if socket is None:
            msg = 'WeCom connection is unavailable.'
            raise WeComClientError(msg)
        request_id = uuid.uuid4().hex
        frame: dict[str, Any] = {'cmd': command, 'headers': {'req_id': request_id}}
        if body is not None:
            frame['body'] = body

        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            async with self._send_lock:
                await socket.send(json.dumps(frame, ensure_ascii=False))
            response = await asyncio.wait_for(future, timeout=20)
        finally:
            self._pending.pop(request_id, None)

        if response.get('errcode') != 0:
            code = response.get('errcode')
            detail = response.get('errmsg')
            message = f'{command} failed: errcode={code}, errmsg={detail}'
            raise WeComClientError(message)
        return response

    def _fail_pending(self, error: Exception) -> None:
        pending, self._pending = self._pending, {}
        for future in pending.values():
            if not future.done():
                future.set_exception(error)
