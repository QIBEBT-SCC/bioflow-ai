import shutil
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path as FilePath
from typing import Any

from anyio import Path
from langchain_core.messages import HumanMessage
from loguru import logger
from sqlalchemy.orm import Session

from bioflow_ai.core.config import get_settings
from bioflow_ai.llm.observability.execution_logging import AgentRunLogger
from bioflow_ai.llm.observability.tracing import configure_langsmith_tracing
from bioflow_ai.services.agents.execution import consume_agent_graph
from bioflow_ai.services.progress import ProgressCallback, emit_progress

DB_PATH = 'presets/build/evaluate.sqlite3'
MISSING_PROJECT_TAG = 'missing'


@dataclass
class WorkflowGenerationResult:
    workflow_uids: list[str]
    new_tool_uids: list[str]
    new_db_ids: list[int]


class WorkflowStateUnavailableError(RuntimeError):
    """Raised when the workflow builder checkpoint cannot be read after execution."""


class WorkflowResumeUnavailableError(RuntimeError):
    """Raised when no interrupted workflow builder checkpoint can be resumed."""


class WorkflowResumeRequiredError(RuntimeError):
    """Raised when a paused workflow checkpoint requires an explicit resume value."""


class EvaluationUsageError(ValueError):
    """Raised when evaluation options cannot be combined."""


def _get_interrupt_question(interrupt_value: Any) -> str:
    """Return the question carried by a workflow-builder interrupt."""
    if not isinstance(interrupt_value, dict):
        message = f'Workflow interrupt payload must be an object, got {type(interrupt_value).__name__}.'
        raise TypeError(message)

    question = interrupt_value.get('question')
    if not isinstance(question, str) or not question.strip():
        message = 'Workflow interrupt payload does not contain a non-empty question.'
        raise ValueError(message)
    return question


def _update_interrupted_project(session: Session, project_id: int, question: str) -> None:
    """Mark an evaluate project as missing and replace its description with the interrupt question."""
    from sqlalchemy import select

    from bioflow_ai.models.project import Project, ProjectTag
    from bioflow_ai.utils.system import utc_now

    project = session.get(Project, project_id)
    if project is None:
        message = f'Evaluate project {project_id} no longer exists.'
        raise RuntimeError(message)

    missing_tag = session.scalar(select(ProjectTag).where(ProjectTag.name == MISSING_PROJECT_TAG))
    if missing_tag is None:
        message = f'Project tag {MISSING_PROJECT_TAG!r} is not configured.'
        raise RuntimeError(message)

    project.description = question
    if all(tag.id != missing_tag.id for tag in project.tags):
        project.tags.append(missing_tag)
    project.update_time = utc_now()
    session.commit()


def _mark_project_interrupted(project_id: int, interrupt_value: Any) -> str:
    """Persist workflow interrupt metadata on its evaluate project."""
    from bioflow_ai.infrastructure.database import engine

    question = _get_interrupt_question(interrupt_value)
    with Session(engine) as session:
        _update_interrupted_project(session, project_id, question)
    logger.info(f'Marked evaluate project {project_id} with tag {MISSING_PROJECT_TAG!r}.')
    return question


def _prepare_workflow_resume_input(state: Any, resume: str) -> Any:
    """Build the LangGraph input required to continue an incomplete checkpoint."""
    from langgraph.types import Command

    interrupts = state.interrupts
    if interrupts:
        logger.info(f'Resuming workflow builder from {len(interrupts)} interrupt(s).')
        return Command(resume=resume)

    if state.next:
        logger.info(f'Resuming incomplete workflow builder checkpoint at: {", ".join(state.next)}.')
        return None

    message = (
        'The workflow builder checkpoint has no pending work to resume. '
        'Run without --resume to start normally, or use --clean-checkpoint to rebuild it.'
    )
    raise WorkflowResumeUnavailableError(message)


def _prepare_workflow_stream_input(
    state: Any,
    agent_input: dict[str, Any],
    *,
    resume: str | None,
) -> Any:
    """Choose a safe input for a new workflow-builder stream.

    LangGraph checkpoints the model response before executing its tool calls. If a process
    stops at that boundary, appending fresh human messages leaves the old tool call without a
    matching ToolMessage and OpenAI rejects the resulting history. Resume ordinary pending
    graph work in place, while requiring an explicit value for human interrupts.
    """
    if resume is not None:
        return _prepare_workflow_resume_input(state, resume)

    if state.interrupts:
        message = (
            f'The workflow builder checkpoint is paused at {len(state.interrupts)} interrupt(s) and cannot accept '
            'new input safely. Use --resume MESSAGE to respond to it, or --clean-checkpoint to discard the '
            'checkpoint and rebuild the workflow.'
        )
        raise WorkflowResumeRequiredError(message)

    if state.next:
        logger.info(f'Automatically resuming incomplete workflow builder checkpoint at: {", ".join(state.next)}.')
        return None

    return agent_input


def _resolve_thread_id(task_file: str, thread_id: str | None) -> str:
    if thread_id:
        return thread_id

    path = FilePath(task_file)
    return path.parent.name if path.name == 'task.md' else path.name


def _check_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    FilePath(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA busy_timeout=30000')
    con.execute("""
        CREATE TABLE IF NOT EXISTS evaluate_results (
            thread_id    TEXT PRIMARY KEY,
            task_prompt  TEXT NOT NULL,
            paper_file   TEXT,
            project_id   INTEGER,
            sample_uid   TEXT,
            workflow_uid TEXT,
            output_path  TEXT,
            new_tool_uids TEXT,
            new_db_ids   TEXT
        )
    """)

    return con


def _serialize_ids(values: list[Any]) -> str:
    return ';'.join(str(value) for value in values)


def _parse_int_ids(raw_ids: str | None) -> list[int]:
    if not raw_ids:
        return []
    return _normalize_int_values([value.strip() for value in raw_ids.split(';') if value.strip()])


def _persist_generated_records(
    con: sqlite3.Connection,
    thread_id: str,
    tool_uids: list[str],
    db_ids: list[int],
) -> None:
    """Merge generated tool/database IDs into the existing evaluate result columns."""
    row = con.execute(
        'SELECT new_tool_uids, new_db_ids FROM evaluate_results WHERE thread_id = :thread_id',
        {'thread_id': thread_id},
    ).fetchone()
    if row:
        historical_tools = list(dict.fromkeys([*_parse_workflow_uids(row['new_tool_uids']), *tool_uids]))
        historical_dbs = list(dict.fromkeys([*_parse_int_ids(row['new_db_ids']), *db_ids]))
        con.execute(
            """UPDATE evaluate_results SET new_tool_uids = ?, new_db_ids = ? WHERE thread_id = ?""",
            (_serialize_ids(historical_tools), _serialize_ids(historical_dbs), thread_id),
        )


def _normalize_uuid_values(values: list[Any]) -> list[str]:
    result = []
    for value in values:
        try:
            result.append(str(uuid.UUID(str(value))))
        except ValueError:
            logger.warning(f'Ignoring invalid uuid in workflow builder state: {value}')
    return result


def _normalize_int_values(values: list[Any]) -> list[int]:
    result = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            logger.warning(f'Ignoring invalid integer id in workflow builder state: {value}')
    return result


def _get_project_path(project_id: int) -> FilePath:
    return FilePath(get_settings().persistent.storage.STORAGE_PATH) / 'proj' / f'{project_id}'


def _parse_workflow_uids(raw_uids: str | None) -> list[str]:
    if not raw_uids:
        return []

    workflow_uids = []
    for raw_uid in raw_uids.split(';'):
        workflow_uid = raw_uid.strip()
        if not workflow_uid:
            continue
        try:
            workflow_uids.append(str(uuid.UUID(workflow_uid)))
        except ValueError:
            logger.warning(f'Ignoring invalid workflow uid in evaluate record: {workflow_uid}')
    return workflow_uids


async def _clean_project_workflows(con: sqlite3.Connection, thread_id: str, project_id: int) -> None:
    from sqlalchemy import select

    import bioflow_ai.crud.workflow as wf_crud
    import bioflow_ai.services.projects.workflows as proj_wf_service
    from bioflow_ai.infrastructure.database import engine
    from bioflow_ai.llm.vectorstores import MilvusVectorStore
    from bioflow_ai.models.workflow import ProjectWorkflow, Workflow

    cursor = con.execute(
        'SELECT workflow_uid FROM evaluate_results WHERE thread_id = :thread_id',
        {'thread_id': thread_id},
    )
    row = cursor.fetchone()
    generated_workflow_uids = set(_parse_workflow_uids(row['workflow_uid'] if row else None))

    with Session(engine) as session:
        project_workflow_uids = session.scalars(
            select(ProjectWorkflow.workflow_uid).where(ProjectWorkflow.project_id == project_id)
        ).all()

        for workflow_uid in project_workflow_uids:
            if workflow_uid is None:
                continue
            proj_wf_service.remove_workflow_from_project(session, project_id, workflow_uid)

        existing_generated_workflow_uids = session.scalars(
            select(Workflow.uid).where(Workflow.uid.in_([uuid.UUID(uid) for uid in generated_workflow_uids]))
        ).all()
        for workflow_uid in existing_generated_workflow_uids:
            wf_crud.delete(session, str(workflow_uid))
        session.commit()

    vectorstore = MilvusVectorStore(
        collection_name='WorkflowCollection',
        uri=get_settings().knowledge_base.MILVUS_URL,
        hybrid=True,
    )
    async with vectorstore:
        for workflow_uid in existing_generated_workflow_uids:
            await vectorstore.adelete(
                filter_expression='wf_uid == {uid}',
                filter_params={'uid': str(workflow_uid)},
            )

    con.execute(
        """
        UPDATE evaluate_results
        SET workflow_uid = NULL, output_path = NULL
        WHERE thread_id = :thread_id
        """,
        {'thread_id': thread_id},
    )
    logger.info(
        f'Cleaned {len(project_workflow_uids)} project workflow import(s) and '
        f'{len(existing_generated_workflow_uids)} generated workflow record(s) for project {project_id}.'
    )


async def _clean_project_contents(con: sqlite3.Connection, thread_id: str, project_id: int) -> None:
    """Remove evaluate-generated project contents while retaining the project itself."""
    from sqlalchemy import delete, select

    from bioflow_ai.infrastructure.database import engine
    from bioflow_ai.models.project import ProjectFileMapping
    from bioflow_ai.models.resource import Sample, SampleFiles
    from bioflow_ai.models.workflow import WorkflowRun

    await _clean_project_workflows(con, thread_id, project_id)

    with Session(engine) as session:
        session.execute(delete(WorkflowRun).where(WorkflowRun.project_id == project_id))

        sample_uids = session.scalars(select(Sample.uid).where(Sample.project_id == project_id)).all()
        if sample_uids:
            session.execute(delete(SampleFiles).where(SampleFiles.sample_uid.in_(sample_uids)))
        session.execute(delete(Sample).where(Sample.project_id == project_id))
        session.execute(delete(ProjectFileMapping).where(ProjectFileMapping.project_id == project_id))
        session.commit()

    project_path = _get_project_path(project_id)
    shutil.rmtree(project_path, ignore_errors=True)
    project_path.mkdir(parents=True, exist_ok=True)

    con.execute(
        """
        UPDATE evaluate_results
        SET sample_uid = NULL, workflow_uid = NULL, output_path = NULL
        WHERE thread_id = :thread_id
        """,
        {'thread_id': thread_id},
    )
    logger.info(
        f'Cleaned project-scoped evaluate data and files for project {project_id}; retained the project record.'
    )


def _init_project(con: sqlite3.Connection, thread_id: str) -> tuple[int, str, str]:
    import bioflow_ai.services.projects.project as project_service
    from bioflow_ai.infrastructure.database import engine
    from bioflow_ai.models.project import Project
    from bioflow_ai.schemas.project import ProjectCreateProp

    cursor = con.execute(
        'SELECT project_id FROM evaluate_results WHERE thread_id = :thread_id',
        {'thread_id': thread_id},
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
        project_id = row[0]
        with Session(engine) as session:
            project = session.get(Project, project_id)
            project_name = project.name if project is not None else None
        if project_name is not None:
            project_path = _get_project_path(project_id)
            project_path.mkdir(parents=True, exist_ok=True)
            return project_id, project_name, f'{project_path}'
        logger.warning(f'Evaluate record points to missing project {project_id}; creating a new project.')

    with Session(engine) as session:
        task_project = project_service.create_project(
            session,
            ProjectCreateProp(name=f'wf_{thread_id}', public=False),
            owner_id=2,
        )
        task_project_id = task_project.id
        task_project_name = task_project.name

    con.execute(
        """
        INSERT INTO evaluate_results (thread_id, task_prompt, paper_file, project_id)
        VALUES (:thread_id, '', NULL, :project_id)
        ON CONFLICT (thread_id) DO UPDATE SET
            project_id = EXCLUDED.project_id
        """,
        {'thread_id': thread_id, 'project_id': task_project_id},
    )

    project_path = _get_project_path(task_project_id)
    project_path.mkdir(parents=True, exist_ok=True)

    return task_project_id, task_project_name, f'{project_path}'


async def _get_sample_info(
    user_input: HumanMessage,
    project_id: int,
    workdir: str,
    checkpoint: Any,
    *,
    on_event: ProgressCallback | None = None,
) -> str:
    from langchain_core.runnables import RunnableConfig
    from uuid_extensions import uuid7str

    from bioflow_ai.core.config import get_settings
    from bioflow_ai.llm.agents.fs_agent.sample_manager import build_evaluate_sample_agent
    from bioflow_ai.llm.context import ProjectContext

    sample_file = Path(get_settings().persistent.storage.STORAGE_PATH) / 'proj' / f'{project_id}' / 'SAMPLES.md'
    if await sample_file.exists():
        sample_info = await sample_file.read_text(encoding='utf-8')
        logger.info(f'Reusing persisted sample info from {sample_file}.')
        return sample_info

    thread_id = uuid7str()
    logger.info(f'Generating sample info for project {project_id}; checkpoint thread_id={thread_id}.')
    agent = build_evaluate_sample_agent(checkpointer=checkpoint)
    config = RunnableConfig(configurable={'thread_id': thread_id}, recursion_limit=sys.maxsize)
    agent_input = {'messages': [user_input]}

    state = await consume_agent_graph(
        agent,
        agent_input,
        config,
        context=ProjectContext(
            user_id=2,
            project_id=project_id,
            workdir=workdir,
        ),
        execution_log=AgentRunLogger('sample-manager', thread_id, thread_id=thread_id),
        on_chunk=lambda chunk: emit_progress(on_event, 'chunk', chunk),
    )
    # 有中断
    if state.tasks and any(hasattr(t, 'interrupts') and t.interrupts for t in state.tasks):
        for task in state.tasks:
            if hasattr(task, 'interrupts'):
                for interrupt in task.interrupts:
                    emit_progress(on_event, 'interrupt', interrupt)
                    sys.exit(-1)

    sample_info = await sample_file.read_text(encoding='utf-8')
    logger.info(f'Sample info generated at {sample_file} ({len(sample_info)} chars).')
    return sample_info


async def _build_input(
    con: sqlite3.Connection,
    task_file: str,
    paper_file: str | None,
    thread_id: str,
) -> HumanMessage:
    import base64

    from bioflow_ai.llm.utils.multimodal import filter_multimedia_message

    task_path = Path(task_file)
    if not await task_path.exists():
        logger.error('Task file not found!')
        sys.exit(0)

    task_data = {'thread_id': thread_id, 'task_prompt': await task_path.read_text(encoding='utf-8')}

    if paper_file:
        paper_path = Path(paper_file)
        if not await paper_path.exists():
            logger.error('Paper file not found!')
            sys.exit(0)
        task_data['paper_file'] = paper_file

        base64_encoded = base64.b64encode(await paper_path.read_bytes()).decode('utf-8')
        init_human_message = HumanMessage(
            content=[
                {
                    'type': 'text',
                    'text': await task_path.read_text(encoding='utf-8'),
                },
                {
                    'type': 'file',
                    'base64': base64_encoded,
                    'mime_type': 'application/pdf',
                    'filename': paper_path.name,
                },
            ]
        )
        human_input = await filter_multimedia_message(init_human_message)
    else:
        human_input = HumanMessage(
            content=[
                {
                    'type': 'text',
                    'text': await task_path.read_text(encoding='utf-8'),
                },
                {
                    'type': 'text',
                    'text': 'The above is all the information and requirements I can provide. The rest is entirely up to you. Do not ask me for any additional information; I have nothing more to tell you.',
                },
            ]
        )

    task_data.setdefault('paper_file', None)
    con.execute(
        """
        INSERT INTO evaluate_results (thread_id, task_prompt, paper_file)
        VALUES (:thread_id, :task_prompt, :paper_file)
        ON CONFLICT (thread_id) DO UPDATE SET
            task_prompt = EXCLUDED.task_prompt,
            paper_file  = EXCLUDED.paper_file
        """,
        task_data,
    )
    return human_input


async def _generate_plan(
    human_input: list[HumanMessage],
    project_id: int,
    workdir: str,
    checkpoint: Any,
    *,
    on_event: ProgressCallback | None = None,
) -> str:
    from langchain_core.runnables import RunnableConfig
    from uuid_extensions import uuid7str

    from bioflow_ai.core.config import get_settings
    from bioflow_ai.llm.agents.workflow_agent.workflow_designer_v2 import build_evaluate_workflow_planner
    from bioflow_ai.llm.context import ProjectContext

    plan_file = Path(get_settings().persistent.storage.STORAGE_PATH) / 'proj' / f'{project_id}' / 'PLAN.md'
    if await plan_file.exists():
        plan = await plan_file.read_text(encoding='utf-8')
        logger.info(f'Reusing persisted plan from {plan_file}.')
        return plan

    thread_id = uuid7str()
    logger.info(f'Generating plan for project {project_id}; checkpoint thread_id={thread_id}.')
    agent = build_evaluate_workflow_planner(checkpointer=checkpoint)
    config = RunnableConfig(configurable={'thread_id': thread_id}, recursion_limit=sys.maxsize)
    agent_input = {'messages': human_input}

    await consume_agent_graph(
        agent,
        agent_input,
        config,
        context=ProjectContext(user_id=2, project_id=project_id, workdir=workdir, auto_approve=True),
        execution_log=AgentRunLogger('workflow-planner', thread_id, thread_id=thread_id),
        on_chunk=lambda chunk: emit_progress(on_event, 'chunk', chunk),
    )

    plan = await plan_file.read_text(encoding='utf-8')
    logger.info(f'Plan generated at {plan_file} ({len(plan)} chars).')
    return plan


async def _generate_workflow(
    con: sqlite3.Connection,
    human_input: list[Any],
    thread_id: str,
    project_id: int,
    workdir: str,
    checkpoint: Any,
    *,
    reset_checkpoint: bool = False,
    resume: str | None = None,
    on_event: ProgressCallback | None = None,
) -> WorkflowGenerationResult:
    from langchain_core.runnables import RunnableConfig

    from bioflow_ai.llm.agents.workflow_agent.workflow_builder import build_workflow_builder
    from bioflow_ai.llm.context import ProjectContext
    from bioflow_ai.llm.utils.session import delete_checkpoint

    async def persist_state() -> tuple[Any, WorkflowGenerationResult]:
        state = await agent.aget_state(config)
        result = WorkflowGenerationResult(
            workflow_uids=_normalize_uuid_values(state.values.get('workflow_uids') or []),
            new_tool_uids=_normalize_uuid_values(state.values.get('new_tools') or []),
            new_db_ids=_normalize_int_values(state.values.get('new_dbs') or []),
        )
        _persist_generated_records(con, thread_id, result.new_tool_uids, result.new_db_ids)
        return state, result

    agent = build_workflow_builder(checkpointer=checkpoint)
    config = RunnableConfig(configurable={'thread_id': thread_id}, recursion_limit=sys.maxsize)
    agent_input = {'messages': human_input}

    if reset_checkpoint:
        # The checkpoint may be disposable, but tools/databases created by that run are not.
        try:
            await persist_state()
        except Exception as exc:
            logger.warning(f'Unable to persist generated records from the old checkpoint: {exc}')
        await delete_checkpoint(checkpoint, thread_id)

    saved_state = await agent.aget_state(config)
    stream_input = _prepare_workflow_stream_input(
        saved_state,
        agent_input,
        resume=resume,
    )

    state = None
    result = WorkflowGenerationResult([], [], [])
    try:
        await consume_agent_graph(
            agent,
            stream_input,
            config,
            context=ProjectContext(
                user_id=2,
                project_id=project_id,
                workdir=workdir,
                auto_approve=True,
            ),
            execution_log=AgentRunLogger('workflow-builder', thread_id, thread_id=thread_id),
            on_chunk=lambda chunk: emit_progress(on_event, 'chunk', chunk),
            resumed=stream_input is None or resume is not None,
        )
    finally:
        # This also runs for agent errors, asyncio cancellation, Ctrl-C and LangGraph interrupts.
        # Checkpoints are written during streaming, so the latest state contains IDs created so far.
        try:
            state, result = await persist_state()
        except Exception as exc:
            logger.exception(f'Unable to persist generated records from workflow state: {exc}')

    if state is None:
        raise WorkflowStateUnavailableError
    # 有中断
    if state.tasks and any(hasattr(t, 'interrupts') and t.interrupts for t in state.tasks):
        for task in state.tasks:
            if hasattr(task, 'interrupts'):
                for interrupt in task.interrupts:
                    _mark_project_interrupted(project_id, interrupt.value)
                    emit_progress(on_event, 'interrupt', interrupt)
                    sys.exit(-1)

    logger.success(f'Workflows generated: {result.workflow_uids}')
    if result.new_tool_uids:
        logger.success(f'Tools created during this build: {result.new_tool_uids}')
    if result.new_db_ids:
        logger.success(f'Databases created during this build: {result.new_db_ids}')

    return result


def _log_created_records(result: WorkflowGenerationResult) -> None:
    if not result.new_tool_uids and not result.new_db_ids:
        logger.info('No tools or databases were created during this build.')
        return

    import bioflow_ai.crud.resource.bio_db as db_crud
    import bioflow_ai.crud.tools.tool as tool_crud
    from bioflow_ai.infrastructure.database import engine

    with Session(engine) as session:
        if result.new_tool_uids:
            tool_labels = []
            for tool_uid in result.new_tool_uids:
                tool = tool_crud.get(session, tool_uid)
                tool_labels.append(f'{tool.name}({tool_uid})' if tool else tool_uid)
            logger.success(f'Created tools recorded: {", ".join(tool_labels)}')

        if result.new_db_ids:
            db_labels = []
            for db_id in result.new_db_ids:
                db = db_crud.get(session, db_id)
                if db is None:
                    db_labels.append(str(db_id))
                else:
                    db_labels.append(f'{db.name}({db_id})')
            logger.success(f'Created databases recorded: {", ".join(db_labels)}')


def _send_run_task(
    con: sqlite3.Connection,
    thread_id: str,
    project_id: int,
    result: WorkflowGenerationResult,
    output_dir: str,
) -> None:
    import bioflow_ai.crud.project_workflow as proj_wf_crud
    import bioflow_ai.services.projects.workflows as proj_wf_service
    from bioflow_ai.infrastructure.database import engine

    con.execute(
        """
        UPDATE evaluate_results
        SET workflow_uid = :workflow_uid,
            output_path = :output_path
        WHERE thread_id = :thread_id
        """,
        {
            'thread_id': thread_id,
            'workflow_uid': _serialize_ids(result.workflow_uids),
            'output_path': output_dir,
        },
    )
    _persist_generated_records(con, thread_id, result.new_tool_uids, result.new_db_ids)
    _log_created_records(result)

    with Session(engine) as session:
        for uid in result.workflow_uids:
            workflow_uid = uuid.UUID(uid)
            if proj_wf_crud.exists(session, project_id, workflow_uid):
                logger.info(f'Workflow {workflow_uid} is already imported into project {project_id}; skipping.')
                continue
            proj_wf_service.add_workflow_to_project(session, project_id, workflow_uid)


async def wf_eval(
    task_file: str,
    paper_file: str | None,
    thread_id: str | None,
    *,
    drop_old: bool,
    clean_checkpoint: bool = False,
    clean_all: bool = False,
    resume: str | None = None,
    db_index: int | None = None,
    langsmith_tracing: bool = False,
    on_event: ProgressCallback | None = None,
) -> None:
    """运行bioflow ai 测试."""
    from langgraph.checkpoint.redis import AsyncRedisSaver
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

    configure_langsmith_tracing('bioflow-ai-eval', enabled=langsmith_tracing)

    if db_index is not None:
        logger.warning('--db-index is ignored because evaluate records now use a single SQLite database.')

    if resume is not None and (drop_old or clean_checkpoint or clean_all):
        message = '--resume cannot be combined with checkpoint cleanup options.'
        raise EvaluationUsageError(message)

    thread_id = _resolve_thread_id(task_file, thread_id)
    con = _check_db()
    project_id: int | None = None
    project_name: str | None = None
    try:
        if drop_old:
            logger.warning('--drop-old is deprecated; use --clean-checkpoint instead.')
        reset_checkpoint = drop_old or clean_checkpoint or clean_all

        user_input = await _build_input(con, task_file, paper_file, thread_id)
        project_id, project_name, workdir = _init_project(con, thread_id)
        if clean_all:
            await _clean_project_contents(con, thread_id, project_id)
        elif reset_checkpoint:
            await _clean_project_workflows(con, thread_id, project_id)
        async with AsyncRedisSaver.from_conn_string(
            get_settings().persistent.cache.LLM_CACHE.unicode_string()
        ) as checkpoint:
            checkpoint.serde = JsonPlusRedisSerializer(
                allowed_msgpack_modules=[
                    ('bioflow_ai.schemas.tool_config', 'FileMount'),
                    ('bioflow_ai.schemas.tool_config', 'ImageConfig'),
                    ('bioflow_ai.schemas.tool_config', 'ParamDefine'),
                    ('bioflow_ai.llm.states', 'BuildEdge'),
                    ('bioflow_ai.llm.states', 'BuildNode'),
                    ('bioflow_ai.llm.states', 'WorkflowDraft'),
                    ('bioflow_ai.llm.tools.test_run_report', 'TestRunReport'),
                    ('bioflow_ai.llm.middleware.todo', 'Todo'),
                    ('bioflow_ai.llm.utils.uid_mapper', 'BidirectionalMapper'),
                ]
            )

            sample_info = await _get_sample_info(user_input, project_id, workdir, checkpoint, on_event=on_event)
            standard_plan = await _generate_plan(
                [user_input, HumanMessage(content=sample_info)], project_id, workdir, checkpoint, on_event=on_event
            )

            agent_input = [
                HumanMessage(content=sample_info),
                HumanMessage(content=standard_plan),
                HumanMessage(
                    content='Please complete the task of constructing a work roadmap based on the above information.'
                ),
            ]
            workflow_result = await _generate_workflow(
                con,
                agent_input,
                thread_id,
                project_id,
                workdir,
                checkpoint,
                reset_checkpoint=reset_checkpoint,
                resume=resume,
                on_event=on_event,
            )
        _send_run_task(con, thread_id, project_id, workflow_result, '')
    finally:
        if project_id is not None and project_name is not None:
            logger.info(f'Evaluate project: {project_name} (ID: {project_id}).')
        con.close()
