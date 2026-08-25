import asyncio
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

from celery import Task
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
from langgraph.types import Command
from loguru import logger
from redis.asyncio import Redis
from sqlalchemy.orm import Session

import app.crud.agent as agent_crud
import app.crud.run_instance as run_instance_crud
from app.core.config import get_settings
from app.core.session import engine
from app.models.agent import AgentMessage, AgentRun
from app.schemas.agent import (
    TERMINAL_AGENT_RUN_STATUSES,
    AgentEventType,
    AgentName,
    AgentRunStatus,
)
from app.services import agent as agent_service
from app.services import workflow_fix as workflow_fix_service
from app.services.agent_executor import execute_agent_graph
from app.services.agent_progress import event_payload, result_payload
from app.utils.system import utc_now
from llm.context import CreateToolContext, ProjectContext, WorkflowDiagnosisContext, WorkflowFixContext
from llm.prebuild.dispatch import AGENTS, RouteTask
from script.worker import celery_app

HEARTBEAT_INTERVAL_SECONDS = 10
AGENT_EVENT_CHANNEL_PREFIX = 'bioflow:agent-run:'
MISSING_INPUT_ERROR = 'Agent run input message is missing.'
MISSING_WORKFLOW_RUN_TARGET_ERROR = 'Workflow diagnosis or fix target is missing.'
MISSING_WORKFLOW_RUN_SOURCE_ERROR = 'Workflow diagnosis or fix source run no longer exists.'


def _checkpoint_serializer() -> JsonPlusRedisSerializer:
    return JsonPlusRedisSerializer(
        allowed_msgpack_modules=[
            ('app.schemas.tool_config', 'FileMount'),
            ('app.schemas.tool_config', 'ImageConfig'),
            ('app.schemas.tool_config', 'ParamDefine'),
            ('llm.states', 'BuildEdge'),
            ('llm.states', 'BuildNode'),
            ('llm.states', 'WorkflowDraft'),
            ('llm.tools.test_run_report', 'TestRunReport'),
            ('llm.utils.middleware.todo', 'Todo'),
            ('llm.utils.uid_mapper', 'BidirectionalMapper'),
        ]
    )


def _text_from_parts(message: AgentMessage) -> str:
    return ''.join(str(part.get('text', '')) for part in message.parts if part.get('type') == 'text')


def _load_langchain_messages(session: Session, run: AgentRun) -> tuple[list[BaseMessage], HumanMessage]:
    stored = agent_crud.list_messages(session, run.session_uid)
    history: list[BaseMessage] = []
    current: HumanMessage | None = None
    for message in stored:
        text = _text_from_parts(message)
        if not text:
            continue
        if message.role == 'assistant':
            history.append(AIMessage(content=text, id=str(message.uid)))
        else:
            converted = HumanMessage(content=text, id=str(message.uid))
            history.append(converted)
            if message.run_uid == run.uid and current is None:
                current = converted
    if current is None:
        raise ValueError(MISSING_INPUT_ERROR)
    return history, current


def _message_text(message: BaseMessage | None) -> str:
    if message is None:
        return ''
    if isinstance(message.content, str):
        return message.content
    texts = []
    for part in message.content:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and part.get('type') == 'text':
            texts.append(str(part.get('text', '')))
    return ''.join(texts)


async def _publish_wakeup(redis: Redis, run_uid: uuid.UUID) -> None:
    try:
        await redis.publish(f'{AGENT_EVENT_CHANNEL_PREFIX}{run_uid}', str(utc_now().timestamp()))
    except Exception:
        logger.warning('Unable to publish agent event wakeup for {}; SSE will use polling fallback.', run_uid)


async def _append_event(
    redis: Redis,
    run_uid: uuid.UUID,
    event_type: AgentEventType,
    payload: dict[str, Any],
) -> None:
    with Session(engine) as session:
        run = agent_crud.get_run(session, run_uid)
        if run is None:
            return
        agent_service.append_event(session, run, event_type, payload)
    await _publish_wakeup(redis, run_uid)


async def _transition(
    redis: Redis,
    run_uid: uuid.UUID,
    status: AgentRunStatus,
    event_type: AgentEventType,
    **kwargs: Any,
) -> AgentRun | None:
    with Session(engine) as session:
        run = agent_crud.get_run(session, run_uid)
        if run is None:
            return None
        if run.status == AgentRunStatus.CANCEL_REQUESTED.value and status != AgentRunStatus.CANCELLED:
            status = AgentRunStatus.CANCELLED
            event_type = AgentEventType.CANCELLED
            kwargs = {}
        try:
            updated = agent_service.transition_run(session, run, status, event_type=event_type, **kwargs)
        except agent_service.AgentRunTransitionError:
            session.refresh(run)
            if AgentRunStatus(run.status) in TERMINAL_AGENT_RUN_STATUSES:
                return None
            raise
        session.expunge(updated)
    await _publish_wakeup(redis, run_uid)
    return updated


async def _heartbeat(run_uid: uuid.UUID, agent_task: asyncio.Task[Any]) -> None:
    while not agent_task.done():
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
        with Session(engine) as session:
            run = agent_crud.get_run(session, run_uid)
            if run is None or not agent_service.update_heartbeat(session, run):
                agent_task.cancel()
                return


async def _run_agent(run_uid: uuid.UUID) -> None:
    redis = Redis.from_url(
        get_settings().persistent.cache.API_CACHE.unicode_string(),
        decode_responses=True,
    )
    checkpoint_url = get_settings().persistent.cache.LLM_CACHE.unicode_string()
    try:
        with Session(engine) as session:
            run = agent_crud.get_run(session, run_uid)
            if run is None or run.status != AgentRunStatus.QUEUED.value:
                return
            agent_name = AgentName(run.agent_name)
            owner_id = run.agent_session.owner_id
            project_id = run.project_id
            checkpoint_thread_id = run.checkpoint_thread_id
            resume_payload = (run.result_payload or {}).get('resume')
            history, current_message = _load_langchain_messages(session, run)
            source_run_uid: uuid.UUID | None = None
            source_run_base_dir: str | None = None
            source_workflow_uid: uuid.UUID | None = None
            if agent_name in {AgentName.WORKFLOW_DIAGNOSER, AgentName.WORKFLOW_FIXER}:
                target = agent_crud.get_run_target(session, run.uid)
                if target is None:
                    raise ValueError(MISSING_WORKFLOW_RUN_TARGET_ERROR)  # noqa: TRY301
                source_run = run_instance_crud.get(session, target.source_run_uid)
                if source_run is None:
                    raise ValueError(MISSING_WORKFLOW_RUN_SOURCE_ERROR)  # noqa: TRY301
                source_run_uid = source_run.uid
                source_run_base_dir = source_run.base_dir
                source_workflow_uid = source_run.workflow_uid

        started = await _transition(
            redis,
            run_uid,
            AgentRunStatus.RUNNING,
            AgentEventType.STARTED,
            payload={'agent_name': agent_name.value},
            result_payload=None,
        )
        if started is None:
            return

        async with AsyncRedisSaver.from_conn_string(checkpoint_url) as checkpoint:
            checkpoint.serde = _checkpoint_serializer()
            await checkpoint.asetup()
            spec = AGENTS[cast('RouteTask', agent_name.value)]
            graph = spec.factory(checkpoint)
            config = RunnableConfig(
                configurable={'thread_id': checkpoint_thread_id},
                recursion_limit=2**31 - 1,
            )

            if resume_payload:
                if resume_payload.get('approved'):
                    agent_input: Command | dict[str, Any] = Command(resume=True)
                else:
                    agent_input = Command(resume=str(resume_payload.get('feedback', '')))
            elif agent_name == AgentName.TOOL_GENERATOR:
                agent_input = {
                    'messages': [
                        current_message.model_copy(
                            update={'content': f'<Require>\n{_message_text(current_message)}\n</Require>'}
                        )
                    ]
                }
            elif agent_name == AgentName.WORKFLOW_DIAGNOSER:
                agent_input = {'messages': [current_message]}
            else:
                agent_input = {'messages': history}

            project_path: Path | None = None
            if agent_name == AgentName.TOOL_GENERATOR:
                context: CreateToolContext | ProjectContext | WorkflowDiagnosisContext | WorkflowFixContext = (
                    CreateToolContext(
                        user_id=owner_id,
                        skip_search=True,
                        auto_save=True,
                        test_run_id=str(run_uid),
                    )
                )
            else:
                if project_id is None:
                    error_message = f'{agent_name.value} requires a project.'
                    raise ValueError(error_message)  # noqa: TRY301
                project_path = Path(get_settings().persistent.storage.STORAGE_PATH) / 'proj' / str(project_id)
                resolved_project_path = str(project_path.resolve().expanduser())
                if agent_name == AgentName.WORKFLOW_DIAGNOSER:
                    if source_run_uid is None:
                        raise ValueError(MISSING_WORKFLOW_RUN_TARGET_ERROR)  # noqa: TRY301
                    context = WorkflowDiagnosisContext(
                        user_id=owner_id,
                        project_id=project_id,
                        workdir=resolved_project_path,
                        agent_run_uid=str(run_uid),
                        source_run_uid=str(source_run_uid),
                        workflow_uid=str(source_workflow_uid) if source_workflow_uid else None,
                        run_base_dir=source_run_base_dir,
                    )
                elif agent_name == AgentName.WORKFLOW_FIXER:
                    if source_run_uid is None:
                        raise ValueError(MISSING_WORKFLOW_RUN_TARGET_ERROR)  # noqa: TRY301
                    with Session(engine) as session:
                        fix_run = agent_crud.get_run(session, run_uid)
                        source_run = run_instance_crud.get(session, source_run_uid)
                        if fix_run is None or source_run is None:
                            raise ValueError(MISSING_WORKFLOW_RUN_SOURCE_ERROR)  # noqa: TRY301
                        prepared = workflow_fix_service.prepare_workflow_fix(
                            session,
                            fix_run=fix_run,
                            source_run=source_run,
                            project_root=project_path.resolve().expanduser(),
                        )
                    context = WorkflowFixContext(
                        user_id=owner_id,
                        project_id=project_id,
                        workdir=resolved_project_path,
                        agent_run_uid=str(run_uid),
                        source_run_uid=str(source_run_uid),
                        workflow_uid=str(prepared.workflow_uid),
                        diagnosis_run_uid=str(prepared.diagnosis_run_uid),
                    )
                    if not resume_payload:
                        diagnosis_message = HumanMessage(
                            content=(
                                '<workflow_diagnosis_report>\n'
                                f'{prepared.diagnosis_report}\n'
                                '</workflow_diagnosis_report>\n'
                                'Treat this report as guidance. Inspect the currently loaded workflow before deciding '
                                'which changes, if any, remain necessary.'
                            )
                        )
                        agent_input = {
                            **prepared.state,
                            'messages': [*history[:-1], diagnosis_message, current_message],
                        }
                else:
                    context = ProjectContext(
                        user_id=owner_id,
                        project_id=project_id,
                        workdir=resolved_project_path,
                    )

            async def on_progress(payload: object) -> None:
                await _append_event(
                    redis,
                    run_uid,
                    AgentEventType.PROGRESS,
                    event_payload(payload),
                )

            consumer = asyncio.create_task(
                execute_agent_graph(graph, agent_input, config, context, on_progress),
                name=f'agent-run-{run_uid}',
            )
            heartbeat = asyncio.create_task(_heartbeat(run_uid, consumer), name=f'agent-heartbeat-{run_uid}')
            try:
                execution = await consumer
            except asyncio.CancelledError:
                await _transition(redis, run_uid, AgentRunStatus.CANCELLED, AgentEventType.CANCELLED)
                await checkpoint.adelete_thread(checkpoint_thread_id)
                return
            finally:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat

            if execution.interrupt is not None:
                interrupt = execution.interrupt
                await _transition(
                    redis,
                    run_uid,
                    AgentRunStatus.WAITING_INPUT,
                    AgentEventType.INTERRUPTED,
                    payload={'value': interrupt},
                    interrupt_payload=interrupt,
                )
                return

            messages = execution.values.get('messages', [])
            final_message = next((message for message in reversed(messages) if isinstance(message, AIMessage)), None)
            if final_message is None and messages:
                final_message = messages[-1]
            final_text = _message_text(final_message)

            with Session(engine) as session:
                run = agent_crud.get_run(session, run_uid)
                if run is None:
                    return
                agent_service.import_generated_workflows(session, run, execution.values)
                if final_text:
                    agent_service.add_assistant_message(session, run, final_text)
                result, changes = result_payload(run, execution.values)

            for change in changes:
                await _append_event(redis, run_uid, AgentEventType.RESOURCE_CHANGED, change)
            await _transition(
                redis,
                run_uid,
                AgentRunStatus.COMPLETED,
                AgentEventType.COMPLETED,
                payload={'message': final_text, 'result': result},
                result_payload=result,
            )
            await checkpoint.adelete_thread(checkpoint_thread_id)
    except Exception as error:
        logger.exception('Agent run {} failed.', run_uid)
        await _transition(
            redis,
            run_uid,
            AgentRunStatus.FAILED,
            AgentEventType.FAILED,
            payload={'code': type(error).__name__, 'message': str(error)},
            error_code=type(error).__name__,
            error_message=str(error),
        )
    finally:
        await redis.aclose()


@celery_app.task(bind=True, acks_late=False, name='app.tasks.agent_run.execute_agent_run')
def execute_agent_run_task(self: Task, run_uid: str) -> None:  # noqa: ARG001
    """Execute one durable web-agent run outside the HTTP request lifecycle."""
    asyncio.run(_run_agent(uuid.UUID(run_uid)))
