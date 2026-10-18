import asyncio
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from celery import Task
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from loguru import logger
from redis.asyncio import Redis
from sqlalchemy.orm import Session

import bioflow_ai.crud.agent as agent_crud
from bioflow_ai.core.config import get_settings
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.llm.checkpoint import open_checkpointer
from bioflow_ai.llm.context import CreateToolContext, ProjectContext, WorkflowAssistantContext, WorkflowFixContext
from bioflow_ai.llm.observability.execution_logging import AgentRunLogger
from bioflow_ai.llm.prebuild.dispatch import AGENTS
from bioflow_ai.models.agent import AgentRun
from bioflow_ai.models.workflow import WorkflowRun
from bioflow_ai.runtime.celery_app import celery_app
from bioflow_ai.schemas.agent import TERMINAL_AGENT_RUN_STATUSES, AgentEventType, AgentName, AgentRunStatus
from bioflow_ai.services.agents import sessions as agent_service
from bioflow_ai.services.agents.execution import execute_agent_graph
from bioflow_ai.services.agents.messages import load_langchain_messages, wrap_request
from bioflow_ai.services.agents.progress import event_payload, result_payload
from bioflow_ai.services.agents.workflow_assistant import prepare_workflow_assistant_input
from bioflow_ai.services.workflow import fix as workflow_fix_service
from bioflow_ai.services.workflow.diagnosis import load_latest_diagnosis_report
from bioflow_ai.utils.system import utc_now

HEARTBEAT_INTERVAL_SECONDS = 10
AGENT_EVENT_CHANNEL_PREFIX = 'bioflow:agent-run:'
MISSING_WORKFLOW_RUN_TARGET_ERROR = 'Workflow assistant or fix target is missing.'
MISSING_WORKFLOW_RUN_SOURCE_ERROR = 'Workflow assistant or fix source run no longer exists.'


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
        resumed = bool((run.result_payload or {}).get('resume'))
        execution_log = AgentRunLogger(
            run.agent_name,
            str(run.uid),
            session_id=str(run.session_uid),
            thread_id=run.checkpoint_thread_id,
        )
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
    execution_log.status(
        'resumed'
        if status == AgentRunStatus.RUNNING and resumed
        else {
            AgentRunStatus.RUNNING: 'started',
            AgentRunStatus.WAITING_INPUT: 'waiting_input',
        }.get(status, status.value),
        kwargs.get('error_message')
        or (kwargs.get('payload', {}).get('value') if status == AgentRunStatus.WAITING_INPUT else ''),
    )
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


async def _run_agent(run_uid: uuid.UUID, language: str | None = None) -> None:
    execution_log = AgentRunLogger('agent', str(run_uid))
    redis = Redis.from_url(
        get_settings().persistent.cache.API_CACHE.unicode_string(),
        decode_responses=True,
    )
    try:
        with Session(engine) as session:
            run = agent_crud.get_run(session, run_uid)
            if run is None or run.status != AgentRunStatus.QUEUED.value:
                return
            agent_name = AgentName(run.agent_name)
            owner_id = run.agent_session.owner_id
            project_id = run.project_id
            checkpoint_thread_id = run.checkpoint_thread_id
            execution_log = AgentRunLogger(
                agent_name.value,
                str(run_uid),
                session_id=str(run.session_uid),
                thread_id=checkpoint_thread_id,
            )
            resume_payload = (run.result_payload or {}).get('resume')
            history, current_message = load_langchain_messages(session, run)
            source_run_uid: uuid.UUID | None = None
            source_run_base_dir: str | None = None
            source_workflow_uid: uuid.UUID | None = None
            source_generation: int | None = None
            if agent_name in {AgentName.WORKFLOW_ASSISTANT, AgentName.WORKFLOW_FIXER}:
                target = agent_crud.get_run_target(session, run.uid)
                if target is None:
                    raise ValueError(MISSING_WORKFLOW_RUN_TARGET_ERROR)  # noqa: TRY301
                source_run = session.get(WorkflowRun, target.source_run_uid)
                if source_run is None:
                    raise ValueError(MISSING_WORKFLOW_RUN_SOURCE_ERROR)  # noqa: TRY301
                source_run_uid = source_run.uid
                source_run_base_dir = source_run.base_dir
                source_workflow_uid = source_run.workflow_uid
                source_generation = source_run.generation
                attached_node = current_message.additional_kwargs.get('workflow_node')
                if attached_node is not None and attached_node['generation'] != source_generation:
                    raise ValueError(agent_service.NODE_STALE_ERROR)  # noqa: TRY301

        started = await _transition(
            redis,
            run_uid,
            AgentRunStatus.RUNNING,
            AgentEventType.STARTED,
            payload={'agent_name': agent_name.value},
            result_payload=None,
        )
        if started is None or started.status != AgentRunStatus.RUNNING.value:
            return

        async with open_checkpointer() as checkpoint:
            spec = AGENTS[agent_name.value]
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
                        message.model_copy(
                            update={'content': wrap_request(message.content, '<Require>\n', '\n</Require>')}
                        )
                        if message is current_message
                        else message
                        for message in history
                    ]
                }
            else:
                agent_input = {'messages': history}

            project_path: Path | None = None
            if agent_name == AgentName.TOOL_GENERATOR:
                context: CreateToolContext | ProjectContext | WorkflowAssistantContext | WorkflowFixContext = (
                    CreateToolContext(
                        user_id=owner_id,
                        language=language,
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
                if agent_name == AgentName.WORKFLOW_ASSISTANT:
                    if source_run_uid is None:
                        raise ValueError(MISSING_WORKFLOW_RUN_TARGET_ERROR)  # noqa: TRY301
                    context = WorkflowAssistantContext(
                        user_id=owner_id,
                        language=language,
                        project_id=project_id,
                        workdir=resolved_project_path,
                        agent_run_uid=str(run_uid),
                        source_run_uid=str(source_run_uid),
                        workflow_uid=str(source_workflow_uid) if source_workflow_uid else None,
                        run_base_dir=source_run_base_dir,
                        source_generation=source_generation,
                    )
                    if not resume_payload:
                        with Session(engine) as session:
                            diagnosis = load_latest_diagnosis_report(
                                session,
                                session_uid=started.session_uid,
                                source_run_uid=source_run_uid,
                                project_root=project_path,
                            )
                        agent_input = prepare_workflow_assistant_input(history, current_message, context, diagnosis)
                elif agent_name == AgentName.WORKFLOW_FIXER:
                    if source_run_uid is None:
                        raise ValueError(MISSING_WORKFLOW_RUN_TARGET_ERROR)  # noqa: TRY301
                    with Session(engine) as session:
                        fix_run = agent_crud.get_run(session, run_uid)
                        source_run = session.get(WorkflowRun, source_run_uid)
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
                        language=language,
                        project_id=project_id,
                        workdir=resolved_project_path,
                        agent_run_uid=str(run_uid),
                        source_run_uid=str(source_run_uid),
                        workflow_uid=str(prepared.workflow_uid),
                        diagnosis_run_uid=str(prepared.diagnosis_run_uid) if prepared.diagnosis_run_uid else None,
                        source_generation=source_generation,
                    )
                    if not resume_payload:
                        fix_messages = history
                        if prepared.diagnosis_report is not None:
                            changed_node_ids = {
                                original_id: editing_id
                                for original_id, editing_id in prepared.node_id_mapping.items()
                                if original_id != editing_id
                            }
                            node_id_mapping_message = ''
                            if changed_node_ids:
                                mapping_lines = '\n'.join(
                                    f'{original_id} -> {editing_id}'
                                    for original_id, editing_id in sorted(changed_node_ids.items())
                                )
                                node_id_mapping_message = (
                                    '<workflow_node_id_mapping>\n'
                                    'The loaded workflow had non-canonical node IDs and was normalized for this editing '
                                    'session. Diagnosis report IDs on the left must be translated to the current panel IDs '
                                    'on the right:\n'
                                    f'{mapping_lines}\n'
                                    '</workflow_node_id_mapping>\n'
                                )
                            diagnosis_message = HumanMessage(
                                content=(
                                    '<workflow_diagnosis_report>\n'
                                    f'{prepared.diagnosis_report}\n'
                                    '</workflow_diagnosis_report>\n'
                                    f'{node_id_mapping_message}'
                                    'Treat this report as guidance. Inspect the currently loaded workflow before deciding '
                                    'which changes, if any, remain necessary.'
                                )
                            )
                            fix_messages = [*history[:-1], diagnosis_message, current_message]
                        if prepared.state.get('source_subgraph_path'):
                            source_path = prepared.state['source_subgraph_path']
                            root_id = prepared.node_id_mapping.get(source_path[0], source_path[0])
                            fix_messages = [
                                *fix_messages[:-1],
                                HumanMessage(
                                    content=(
                                        '<source_subgraph_path>\n'
                                        f'{[root_id, *source_path[1:]]}\n'
                                        'The selected failed run is a ForEach child. The main template is loaded. '
                                        'Enter this subgraph path to repair its source definition; do not copy '
                                        'bound paths from the child run into the template.\n'
                                        '</source_subgraph_path>'
                                    )
                                ),
                                current_message,
                            ]
                        agent_input = {
                            **prepared.state,
                            'messages': fix_messages,
                        }
                else:
                    context = ProjectContext(
                        user_id=owner_id,
                        language=language,
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
                execute_agent_graph(graph, agent_input, config, context, on_progress, execution_log=execution_log),
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
        execution_log.status('execution_error', str(error), error=error)
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


@celery_app.task(bind=True, acks_late=False, name='bioflow_ai.tasks.agent_run.execute_agent_run')
def execute_agent_run_task(self: Task, run_uid: str, language: str | None = None) -> None:  # noqa: ARG001
    """Execute one durable web-agent run outside the HTTP request lifecycle."""
    asyncio.run(_run_agent(uuid.UUID(run_uid), language))
