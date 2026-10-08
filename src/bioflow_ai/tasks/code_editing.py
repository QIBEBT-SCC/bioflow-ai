"""Per-turn Celery scheduling for native CODE editing conversations."""

import asyncio
from contextlib import suppress
from typing import Any

from langchain_core.messages import messages_from_dict, messages_to_dict
from redis.asyncio import Redis

from bioflow_ai.core.config import get_settings
from bioflow_ai.llm.observability.execution_logging import AgentRunLogger
from bioflow_ai.runtime.celery_app import celery_app
from bioflow_ai.schemas.code import CodeNodeType
from bioflow_ai.services.code.coding import CodingAgent, run_coding_turn
from bioflow_ai.services.code.sessions import CodeSessionStore, StaleCodingTurnError


async def execute_turn(
    store: CodeSessionStore,
    session_id: str,
    turn_id: str,
    *,
    agent: CodingAgent | None = None,
) -> None:
    """Claim a delivery, monitor cancellation, and publish only complete results."""
    state = await store.claim(session_id, turn_id)
    if state is None:
        return
    request = state['pending_turn']
    execution_log = AgentRunLogger('simple-coding', turn_id, session_id=session_id, thread_id=session_id)
    execution_log.status('started')

    async def progress(event_type: str, payload: dict[str, Any]) -> None:
        await store.progress(session_id, turn_id, event_type, payload)

    job = asyncio.create_task(
        run_coding_turn(
            request['prompt'],
            node_type=CodeNodeType(state['node_type']),
            user_id=state['owner_id'],
            source=request['source'],
            dependencies=request['dependencies'],
            language=request['language'],
            history=messages_from_dict(state['messages']),
            turn_id=turn_id,
            session_id=session_id,
            on_progress=progress,
            agent=agent,
            execution_log=execution_log,
        )
    )

    async def monitor() -> None:
        try:
            while not job.done():
                if not await store.heartbeat(session_id, turn_id):
                    job.cancel()
                    return
                await asyncio.sleep(1)
        except Exception:
            job.cancel()
            raise

    watcher = asyncio.create_task(monitor())
    try:
        result = await asyncio.wait_for(job, timeout=store.config.TURN_TIMEOUT_SECONDS)
        await store.complete(
            session_id,
            turn_id,
            messages_to_dict(result.messages),
            result.proposal.model_dump() if result.proposal is not None else None,
        )
        execution_log.status('completed')
    except asyncio.CancelledError:
        outcome = await store.fail(session_id, turn_id, 'The coding turn was stopped.', cancelled=True)
        if outcome is not None:
            execution_log.status(outcome)
    except TimeoutError as exc:
        outcome = await store.fail(session_id, turn_id, 'The coding turn exceeded its time limit.')
        if outcome is not None:
            execution_log.status(
                'timed_out' if outcome == 'failed' else outcome,
                'The coding turn exceeded its time limit.',
                error=exc,
            )
    except StaleCodingTurnError:
        # Cancellation can win immediately before completion or the next progress
        # callback. Release that turn; closed and superseded turns remain untouched.
        outcome = await store.fail(session_id, turn_id, 'The coding turn was stopped.', cancelled=True)
        if outcome is not None:
            execution_log.status(outcome)
    except Exception as exc:
        outcome = await store.fail(session_id, turn_id, str(exc) or type(exc).__name__)
        if outcome is not None:
            execution_log.status(outcome, str(exc), error=exc)
    finally:
        watcher.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await watcher
        if not job.done():
            job.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await job


@celery_app.task(name='bioflow_ai.tasks.code_editing.execute_code_turn')
def execute_code_turn(session_id: str, turn_id: str) -> None:
    """Run one turn on the existing agent queue without retaining an idle worker."""

    async def run() -> None:
        redis = Redis.from_url(get_settings().persistent.cache.API_CACHE.unicode_string(), decode_responses=True)
        try:
            await execute_turn(CodeSessionStore(redis), session_id, turn_id)
        finally:
            await redis.aclose()

    asyncio.run(run())
