"""Atomic Redis state and replayable events for temporary CODE conversations."""

import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from langchain_core.messages import HumanMessage, message_to_dict
from redis.asyncio import Redis
from redis.exceptions import WatchError

from bioflow_ai.core.config import CodeEditingConfig, get_settings
from bioflow_ai.schemas.code import CodeNodeType
from bioflow_ai.schemas.code_agent import CodeAgentEvent, CodeAgentSessionPublic, CodeAgentTurnCreate
from bioflow_ai.services.code.workspace import baseline_hash

State = dict[str, Any]
Events = list[tuple[str, dict[str, Any]]]
Mutation = Callable[[State], Events]
PREFIX = 'bioflow:code-editing:'


class CodeSessionError(ValueError):
    """An invalid operation on a temporary CODE conversation."""


class CodeSessionNotFoundError(CodeSessionError):
    """The session expired or does not belong to the requesting user."""


class CodeSessionBusyError(CodeSessionError):
    """The session state prevents this operation."""


class StaleCodingTurnError(CodeSessionError):
    """An expired, cancelled, or superseded task must not publish results."""


class CodeSessionStore:
    """Keep state transitions and their SSE events in one Redis transaction."""

    def __init__(self, redis: Redis, config: CodeEditingConfig | None = None):
        self.redis = redis
        self.config = config if config is not None else get_settings().code_editing

    @staticmethod
    def key(session_id: str) -> str:
        """Return the temporary state key."""
        return PREFIX + session_id

    @staticmethod
    def event_key(session_id: str) -> str:
        """Return the replayable event stream key."""
        return PREFIX + session_id + ':events'

    async def _change(
        self,
        session_id: str,
        mutate: Mutation,
        owner_id: int | None = None,
        *,
        refresh_ttl: bool = True,
    ) -> State:
        key = self.key(session_id)
        for _attempt in range(10):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key)
                    raw = await pipe.get(key)
                    if raw is None:
                        raise CodeSessionNotFoundError
                    state = json.loads(raw)
                    if owner_id is not None and state['owner_id'] != owner_id:
                        raise CodeSessionNotFoundError
                    events = mutate(state)
                    pipe.multi()
                    if any(kind == 'session.closed' for kind, _data in events):
                        # Keep only the close notification alongside the tombstone.
                        # Conversation text, tool output and candidates are discarded.
                        pipe.delete(self.event_key(session_id))
                    for event_type, data in events:
                        state['event_seq'] += 1
                        event_id = f'{state["event_seq"]}-0'
                        pipe.xadd(
                            self.event_key(session_id),
                            {'type': event_type, 'data': json.dumps(data, ensure_ascii=False)},
                            id=event_id,
                            maxlen=10_000,
                        )
                    state['last_event_id'] = f'{state["event_seq"]}-0'
                    ttl = 60 if state['status'] == 'closed' else self.config.SESSION_TTL_SECONDS
                    if refresh_ttl:
                        pipe.set(key, json.dumps(state, ensure_ascii=False), ex=ttl)
                        pipe.expire(self.event_key(session_id), ttl)
                    else:
                        pipe.set(key, json.dumps(state, ensure_ascii=False), keepttl=True)
                    await pipe.execute()
                except WatchError:
                    continue
                else:
                    return state
        msg = 'The coding session is busy; retry the operation.'
        raise CodeSessionBusyError(msg)

    async def create(self, owner_id: int, node_type: CodeNodeType) -> State:
        """Create an idle conversation without starting a worker or creating files."""
        session_id = str(uuid.uuid4())
        state = {
            'id': session_id,
            'owner_id': owner_id,
            'node_type': node_type.value,
            'status': 'ready',
            'turn_id': None,
            'pending_turn': None,
            'messages': [],
            'transcript': [],
            'tools': [],
            'proposal': None,
            'error': None,
            'event_seq': 0,
            'last_event_id': '0-0',
            'worker_at': None,
        }
        await self.redis.set(self.key(session_id), json.dumps(state), ex=self.config.SESSION_TTL_SECONDS, nx=True)
        return await self._change(session_id, lambda _state: [('session.ready', {})])

    @staticmethod
    def public(state: State) -> CodeAgentSessionPublic:
        """Remove trusted runtime and model-history fields from API responses."""
        return CodeAgentSessionPublic.model_validate(state)

    @staticmethod
    def _end_tools(state: State, turn_id: str, status: str) -> Events:
        events: Events = []
        for tool in state['tools']:
            if tool['turn_id'] == turn_id and tool['status'] == 'running':
                tool['status'] = status
                events.append(('tool.updated', dict(tool)))
        return events

    async def get(self, session_id: str, owner_id: int, *, touch: bool = True) -> State:
        """Refresh the client's lease and recover abandoned queued or running turns."""

        def refresh(state: State) -> Events:
            now = time.time()
            stale_worker = state['status'] in {'running', 'cancelling'} and (
                now - (state['worker_at'] or now) > self.config.WORKER_STALE_SECONDS
            )
            stale_queue = state['status'] == 'queued' and (
                now - state['pending_turn']['queued_at'] > self.config.TURN_TIMEOUT_SECONDS
            )
            if not stale_worker and not stale_queue:
                return []
            turn_id = state['turn_id']
            state.update(
                status='ready', pending_turn=None, proposal=None, error='The coding worker stopped responding.'
            )
            return [
                *self._end_tools(state, turn_id, 'failed'),
                ('turn.failed', {'turn_id': turn_id, 'message': state['error']}),
            ]

        return await self._change(session_id, refresh, owner_id, refresh_ttl=touch)

    async def submit(self, session_id: str, owner_id: int, data: CodeAgentTurnCreate) -> State:
        """Atomically reserve exactly one turn using the editor's current baseline."""

        def reserve(state: State) -> Events:
            if state['status'] != 'ready':
                msg = 'Finish the active turn or decide its proposal first.'
                raise CodeSessionBusyError(msg)
            turn_id = str(uuid.uuid4())
            state.update(
                status='queued',
                turn_id=turn_id,
                error=None,
                pending_turn={**data.model_dump(), 'queued_at': time.time()},
            )
            state['transcript'].append({'role': 'user', 'text': data.prompt, 'turn_id': turn_id})
            return [('turn.queued', {'turn_id': turn_id, 'prompt': data.prompt})]

        return await self._change(session_id, reserve, owner_id)

    async def claim(self, session_id: str, turn_id: str) -> State | None:
        """Only the first delivery of the current queued task may execute."""

        def start(state: State) -> Events:
            if state['status'] != 'queued' or state['turn_id'] != turn_id:
                raise StaleCodingTurnError
            if time.time() - state['pending_turn']['queued_at'] > self.config.TURN_TIMEOUT_SECONDS:
                state.update(status='ready', pending_turn=None, error='The coding turn expired in the queue.')
                return [('turn.failed', {'turn_id': turn_id, 'message': state['error']})]
            state.update(status='running', worker_at=time.time())
            return [('turn.started', {'turn_id': turn_id})]

        try:
            state = await self._change(session_id, start)
            return state if state['status'] == 'running' else None
        except (StaleCodingTurnError, CodeSessionNotFoundError):
            return None

    @staticmethod
    def _active(state: State, turn_id: str, *, allow_cancelling: bool = False) -> None:
        statuses = {'running', 'cancelling'} if allow_cancelling else {'running'}
        if state['turn_id'] != turn_id or state['status'] not in statuses:
            raise StaleCodingTurnError

    async def heartbeat(self, session_id: str, turn_id: str) -> bool:
        """Renew the worker lease and report cancellation to its execution monitor."""

        def beat(state: State) -> Events:
            self._active(state, turn_id, allow_cancelling=True)
            state['worker_at'] = time.time()
            return []

        try:
            state = await self._change(session_id, beat)
        except (StaleCodingTurnError, CodeSessionNotFoundError):
            return False
        return state['status'] == 'running'

    async def progress(self, session_id: str, turn_id: str, event_type: str, data: dict[str, Any]) -> None:
        """Publish only events from the currently executing task."""

        def publish(state: State) -> Events:
            self._active(state, turn_id)
            state['worker_at'] = time.time()
            if event_type == 'message.delta':
                transcript = state['transcript']
                if not transcript or transcript[-1]['role'] != 'assistant' or transcript[-1]['turn_id'] != turn_id:
                    transcript.append({'role': 'assistant', 'text': '', 'turn_id': turn_id})
                transcript[-1]['text'] += data['text']
            elif event_type == 'tool.updated':
                tools = state['tools']
                previous = next((tool for tool in tools if tool['call_id'] == data['call_id']), None)
                if previous is None:
                    tools.append({**data, 'turn_id': turn_id})
                else:
                    # End events need not repeat the command in their input.
                    previous.update({key: value for key, value in data.items() if value or key == 'output'})
            return [(event_type, {**data, 'turn_id': turn_id})]

        await self._change(session_id, publish)

    async def complete(
        self,
        session_id: str,
        turn_id: str,
        messages: list[dict[str, Any]],
        proposal: dict[str, Any] | None,
    ) -> None:
        """Commit complete model history and a candidate together, never partial history."""

        def finish(state: State) -> Events:
            self._active(state, turn_id)
            state.update(messages=messages, pending_turn=None, proposal=proposal, error=None)
            state['status'] = 'proposal' if proposal is not None else 'ready'
            events: Events = [('turn.completed', {'turn_id': turn_id, 'status': state['status']})]
            if proposal is not None:
                events.append(('proposal.ready', proposal))
            return events

        await self._change(session_id, finish)

    async def fail(self, session_id: str, turn_id: str, message: str, *, cancelled: bool = False) -> None:
        """Release a failed turn without replacing previously complete model history."""

        def finish(state: State) -> Events:
            if state['turn_id'] != turn_id or state['status'] not in {'queued', 'running', 'cancelling'}:
                raise StaleCodingTurnError
            stopped = cancelled or state.get('cancelled_turn') == turn_id
            state.update(status='ready', pending_turn=None, proposal=None, error=None if stopped else message)
            event = 'turn.cancelled' if stopped else 'turn.failed'
            return [
                *self._end_tools(state, turn_id, 'cancelled' if stopped else 'failed'),
                (event, {'turn_id': turn_id, 'message': message}),
            ]

        try:
            await self._change(session_id, finish)
        except (StaleCodingTurnError, CodeSessionNotFoundError):
            return

    async def cancel(self, session_id: str, owner_id: int) -> State:
        """Cancel queued work immediately or ask the active worker to stop safely."""

        def stop(state: State) -> Events:
            if state['status'] == 'queued':
                state.update(status='ready', pending_turn=None)
                return [('turn.cancelled', {'turn_id': state['turn_id']})]
            if state['status'] == 'running':
                state.update(status='cancelling', cancelled_turn=state['turn_id'])
                return [('turn.cancel_requested', {'turn_id': state['turn_id']})]
            return []

        return await self._change(session_id, stop, owner_id)

    async def decide(
        self,
        session_id: str,
        owner_id: int,
        proposal_id: str,
        decision: str,
        source: str,
        dependencies: list[str],
    ) -> State:
        """Record explicit review; acceptance never writes to the CODE catalog."""

        def review(state: State) -> Events:
            proposal = state['proposal']
            if state['status'] != 'proposal' or proposal is None or proposal['id'] != proposal_id:
                msg = 'Proposal is no longer pending.'
                raise CodeSessionBusyError(msg)
            if decision == 'accept' and proposal['baseline_hash'] != baseline_hash(
                CodeNodeType(state['node_type']),
                source,
                dependencies,
            ):
                msg = 'The editor changed after this proposal was requested. Discard it and submit the latest code.'
                raise CodeSessionBusyError(msg)
            feedback = (
                'accepted into the unsaved editor' if decision == 'accept' else 'discarded; the editor is unchanged'
            )
            state['messages'].append(message_to_dict(HumanMessage(f'Platform review result: proposal {feedback}.')))
            state.update(status='ready', proposal=None)
            return [('proposal.decided', {'proposal_id': proposal_id, 'decision': decision})]

        return await self._change(session_id, review, owner_id)

    async def close(self, session_id: str, owner_id: int) -> State:
        """Leave a short-lived tombstone so late tasks cannot resurrect a conversation."""

        def discard(state: State) -> Events:
            if state['status'] == 'closed':
                return []
            state.update(status='closed', pending_turn=None, messages=[], transcript=[], tools=[], proposal=None)
            return [('session.closed', {})]

        return await self._change(session_id, discard, owner_id)

    async def read_events(self, session_id: str, owner_id: int, after_id: str) -> AsyncIterator[CodeAgentEvent]:
        """Replay committed events, then follow new events while renewing the client lease."""
        cursor = after_id
        while True:
            state = await self.get(session_id, owner_id, touch=False)
            rows = await self.redis.xread({self.event_key(session_id): cursor}, count=100, block=1000)
            if not rows:
                if state['status'] == 'closed':
                    return
                yield CodeAgentEvent(id=cursor, type='heartbeat', data={})
                continue
            for _stream, events in rows:
                for event_id, fields in events:
                    cursor = str(event_id)
                    yield CodeAgentEvent(id=cursor, type=fields['type'], data=json.loads(fields['data']))
                    if fields['type'] == 'session.closed':
                        return
