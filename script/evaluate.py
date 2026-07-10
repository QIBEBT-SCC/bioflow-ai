import asyncio
import shutil
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path as FilePath
from typing import Any

import click
from anyio import Path
from langchain_core.messages import HumanMessage
from loguru import logger

from app.core.config import get_settings

DB_PATH = 'presets/build/evaluate.sqlite3'


@dataclass
class WorkflowGenerationResult:
    workflow_uids: list[str]
    new_tool_uids: list[str]
    new_db_ids: list[int]


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
    existing_columns = {row['name'] for row in con.execute('PRAGMA table_info(evaluate_results)').fetchall()}
    for column, column_type in {
        'new_tool_uids': 'TEXT',
        'new_db_ids': 'TEXT',
    }.items():
        if column not in existing_columns:
            con.execute(f'ALTER TABLE evaluate_results ADD COLUMN {column} {column_type}')

    return con


def _serialize_ids(values: list[Any]) -> str:
    return ';'.join(str(value) for value in values)


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


def _clean_up_all(con: sqlite3.Connection, thread_id: str) -> None:
    from sqlmodel import Session

    import app.crud.project as proj_crud
    from app.core.session import engine

    cursor = con.execute(
        'SELECT * FROM evaluate_results WHERE thread_id = :thread_id',
        {'thread_id': thread_id},
    )
    row = cursor.fetchone()

    if row and row['project_id']:
        project_path = _get_project_path(row['project_id'])
        with Session(engine) as session:
            proj_crud.delete(session, row['project_id'])
        shutil.rmtree(project_path, ignore_errors=True)

    con.execute('DELETE FROM evaluate_results WHERE thread_id = :thread_id', {'thread_id': thread_id})


def _parse_workflow_uids(raw_uids: str | None) -> list[str]:
    import uuid

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


def _clean_project_workflows(con: sqlite3.Connection, thread_id: str, project_id: int) -> None:
    import uuid

    from sqlmodel import Session, col, select

    import app.crud.workflow as wf_crud
    import app.services.project_workflow as proj_wf_service
    from app.core.session import engine
    from app.models.project import ProjectWorkflow
    from app.models.workflow import Workflow

    cursor = con.execute(
        'SELECT workflow_uid FROM evaluate_results WHERE thread_id = :thread_id',
        {'thread_id': thread_id},
    )
    row = cursor.fetchone()
    generated_workflow_uids = set(_parse_workflow_uids(row['workflow_uid'] if row else None))

    with Session(engine) as session:
        project_workflow_uids = session.exec(
            select(ProjectWorkflow.workflow_uid).where(ProjectWorkflow.project_id == project_id)
        ).all()

        for workflow_uid in project_workflow_uids:
            if workflow_uid is None:
                continue
            proj_wf_service.remove_workflow_from_project(session, project_id, workflow_uid)

        existing_generated_workflow_uids = session.exec(
            select(Workflow.uid).where(col(Workflow.uid).in_([uuid.UUID(uid) for uid in generated_workflow_uids]))
        ).all()
        for workflow_uid in existing_generated_workflow_uids:
            wf_crud.delete(session, str(workflow_uid))

    con.execute(
        """
        UPDATE evaluate_results
        SET workflow_uid = NULL, output_path = NULL, new_tool_uids = NULL, new_db_ids = NULL
        WHERE thread_id = :thread_id
        """,
        {'thread_id': thread_id},
    )
    logger.info(
        f'Cleaned {len(project_workflow_uids)} project workflow import(s) and '
        f'{len(existing_generated_workflow_uids)} generated workflow record(s) for project {project_id}.'
    )


def _init_project(con: sqlite3.Connection, thread_id: str) -> tuple[int, str]:
    from sqlmodel import Session

    import app.crud.project as proj_crud
    from app.core.session import engine
    from app.models.project import Project

    cursor = con.execute(
        'SELECT project_id FROM evaluate_results WHERE thread_id = :thread_id',
        {'thread_id': thread_id},
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
        project_id = row[0]
        with Session(engine) as session:
            project = session.get(Project, project_id)
        if project is not None:
            project_path = _get_project_path(project_id)
            project_path.mkdir(parents=True, exist_ok=True)
            return project_id, f'{project_path}'
        logger.warning(f'Evaluate record points to missing project {project_id}; creating a new project.')

    with Session(engine) as session:
        task_project = Project(name=f'wf_{thread_id}', owner_id=2, public=False)
        task_project = proj_crud.insert(session, task_project)

    con.execute(
        """
        INSERT INTO evaluate_results (thread_id, task_prompt, paper_file, project_id)
        VALUES (:thread_id, '', NULL, :project_id)
        ON CONFLICT (thread_id) DO UPDATE SET
            project_id = EXCLUDED.project_id
        """,
        {'thread_id': thread_id, 'project_id': task_project.id},
    )

    project_path = _get_project_path(task_project.id)
    project_path.mkdir(parents=True, exist_ok=True)

    return task_project.id, f'{project_path}'


async def _get_sample_info(user_input: HumanMessage, project_id: int, workdir: str, checkpoint: Any) -> str:
    import rich
    from langchain_core.runnables import RunnableConfig
    from rich.console import Console
    from uuid_extensions import uuid7str

    from app.core.config import get_settings
    from llm.agents.fs_agent.sample_manager import build_create_sample_agent
    from llm.context import ProjectContext
    from llm.utils.render import print_stream_chunk

    sample_file = Path(get_settings().persistent.storage.STORAGE_PATH) / 'proj' / f'{project_id}' / 'SAMPLES.md'
    if await sample_file.exists():
        sample_info = await sample_file.read_text(encoding='utf-8')
        logger.info(f'Reusing persisted sample info from {sample_file}.')
        return sample_info

    console = Console()

    thread_id = uuid7str()
    logger.info(f'Generating sample info for project {project_id}; checkpoint thread_id={thread_id}.')
    agent = build_create_sample_agent(checkpointer=checkpoint)
    config = RunnableConfig(configurable={'thread_id': thread_id}, recursion_limit=sys.maxsize)
    agent_input = {'messages': [user_input]}

    # pyrefly: ignore [no-matching-overload]
    async for chunk in agent.astream(
        agent_input,
        config,
        context=ProjectContext(user_id=2, project_id=project_id, workdir=workdir),
        stream_mode=['updates', 'custom'],
        version='v2',
        subgraphs=True,
    ):
        print_stream_chunk(console, chunk)

    state = await agent.aget_state(config)
    # 有中断
    if state.tasks and any(hasattr(t, 'interrupts') and t.interrupts for t in state.tasks):
        for task in state.tasks:
            if hasattr(task, 'interrupts'):
                for interrupt in task.interrupts:
                    rich.print(interrupt)
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

    from llm.utils.multimodal import filter_multimedia_message

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

    # pyrefly: ignore [no-matching-overload]
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


async def _generate_plan(human_input: list[HumanMessage], project_id: int, workdir: str, checkpoint: Any) -> str:
    from langchain_core.runnables import RunnableConfig
    from rich.console import Console
    from uuid_extensions import uuid7str

    from app.core.config import get_settings
    from llm.agents.workflow_agent.workflow_designer_v2 import build_workflow_planner
    from llm.context import ProjectContext
    from llm.utils.render import print_stream_chunk

    plan_file = Path(get_settings().persistent.storage.STORAGE_PATH) / 'proj' / f'{project_id}' / 'PLAN.md'
    if await plan_file.exists():
        plan = await plan_file.read_text(encoding='utf-8')
        logger.info(f'Reusing persisted plan from {plan_file}.')
        return plan

    console = Console()

    thread_id = uuid7str()
    logger.info(f'Generating plan for project {project_id}; checkpoint thread_id={thread_id}.')
    agent = build_workflow_planner(checkpointer=checkpoint)
    config = RunnableConfig(configurable={'thread_id': thread_id}, recursion_limit=sys.maxsize)
    agent_input = {'messages': human_input}

    # pyrefly: ignore [no-matching-overload]
    async for chunk in agent.astream(
        agent_input,
        config,
        context=ProjectContext(user_id=2, project_id=project_id, workdir=workdir, auto_approve=True),
        stream_mode=['updates', 'custom'],
        version='v2',
        subgraphs=True,
    ):
        print_stream_chunk(console, chunk)

    state = await agent.aget_state(config)

    plan = state.values.get('plan', '')
    logger.info(f'Plan generated for project {project_id} ({len(plan)} chars).')
    return plan


async def _generate_workflow(
    human_input: list[Any],
    thread_id: str,
    project_id: int,
    workdir: str,
    checkpoint: Any,
    *,
    reset_checkpoint: bool = False,
) -> WorkflowGenerationResult:
    import rich
    from langchain_core.runnables import RunnableConfig
    from rich.console import Console

    from llm.agents.workflow_agent.workflow_builder import build_workflow_builder
    from llm.context import ProjectContext
    from llm.utils.render import print_stream_chunk
    from llm.utils.session import delete_checkpoint

    console = Console()

    if reset_checkpoint:
        await delete_checkpoint(checkpoint, thread_id)

    agent = build_workflow_builder(checkpointer=checkpoint)
    config = RunnableConfig(configurable={'thread_id': thread_id}, recursion_limit=sys.maxsize)
    agent_input = {'messages': human_input}

    # pyrefly: ignore [no-matching-overload]
    async for chunk in agent.astream(
        agent_input,
        config,
        context=ProjectContext(user_id=2, project_id=project_id, workdir=workdir),
        stream_mode=['updates', 'custom'],
        version='v2',
        subgraphs=True,
    ):
        print_stream_chunk(console, chunk)

    state = await agent.aget_state(config)
    # 有中断
    if state.tasks and any(hasattr(t, 'interrupts') and t.interrupts for t in state.tasks):
        for task in state.tasks:
            if hasattr(task, 'interrupts'):
                for interrupt in task.interrupts:
                    rich.print(interrupt)
                    sys.exit(-1)

    workflow_uids = _normalize_uuid_values(state.values.get('workflow_uids') or [])
    new_tool_uids = _normalize_uuid_values(state.values.get('new_tools') or [])
    new_db_ids = _normalize_int_values(state.values.get('new_dbs') or [])
    logger.success(f'工作流已生成: {workflow_uids}')
    if new_tool_uids:
        logger.success(f'本次构建新增工具: {new_tool_uids}')
    if new_db_ids:
        logger.success(f'本次构建新增数据库: {new_db_ids}')

    return WorkflowGenerationResult(
        workflow_uids=workflow_uids,
        new_tool_uids=new_tool_uids,
        new_db_ids=new_db_ids,
    )


def _log_created_records(result: WorkflowGenerationResult) -> None:
    if not result.new_tool_uids and not result.new_db_ids:
        logger.info('本次构建未新增工具或数据库。')
        return

    from sqlmodel import Session

    import app.crud.resource.bio_db as db_crud
    import app.crud.tools.tool as tool_crud
    from app.core.session import engine

    with Session(engine) as session:
        if result.new_tool_uids:
            tool_labels = []
            for tool_uid in result.new_tool_uids:
                tool = tool_crud.get(session, tool_uid)
                tool_labels.append(f'{tool.name}({tool_uid})' if tool else tool_uid)
            logger.success(f'新增工具已记录: {", ".join(tool_labels)}')

        if result.new_db_ids:
            db_labels = []
            for db_id in result.new_db_ids:
                try:
                    db = db_crud.get(session, db_id)
                    db_labels.append(f'{db.name}({db_id})')
                except Exception:
                    db_labels.append(str(db_id))
            logger.success(f'新增数据库已记录: {", ".join(db_labels)}')


def _send_run_task(
    con: sqlite3.Connection,
    thread_id: str,
    project_id: int,
    result: WorkflowGenerationResult,
    output_dir: str,
) -> None:
    from sqlmodel import Session

    import app.crud.project_workflow as proj_wf_crud
    from app.core.session import engine
    from app.models.user import User  # noqa: F401

    con.execute(
        """
        UPDATE evaluate_results
        SET workflow_uid = :workflow_uid,
            output_path = :output_path,
            new_tool_uids = :new_tool_uids,
            new_db_ids = :new_db_ids
        WHERE thread_id = :thread_id
        """,
        {
            'thread_id': thread_id,
            'workflow_uid': _serialize_ids(result.workflow_uids),
            'output_path': output_dir,
            'new_tool_uids': _serialize_ids(result.new_tool_uids),
            'new_db_ids': _serialize_ids(result.new_db_ids),
        },
    )
    _log_created_records(result)

    with Session(engine) as session:
        for uid in result.workflow_uids:
            workflow_uid = uuid.UUID(uid)
            if proj_wf_crud.is_workflow_in_project(session, project_id, workflow_uid):
                logger.info(f'Workflow {workflow_uid} is already imported into project {project_id}; skipping.')
                continue
            proj_wf_crud.add_workflow_to_project(session, project_id, workflow_uid)


async def wf_eval(
    task_file: str,
    paper_file: str | None,
    thread_id: str | None,
    *,
    drop_old: bool,
    clean_checkpoint: bool = False,
    clean_all: bool = False,
    db_index: int | None = None,
) -> None:
    """运行bioflow ai 测试."""
    import os

    from langgraph.checkpoint.redis import AsyncRedisSaver
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_PROJECT'] = 'bioflow-ai-eval'

    if db_index is not None:
        logger.warning('--db-index is ignored because evaluate records now use a single SQLite database.')

    thread_id = _resolve_thread_id(task_file, thread_id)
    con = _check_db()
    try:
        if drop_old:
            logger.warning('--drop-old is deprecated; use --clean-checkpoint instead.')
        reset_checkpoint = drop_old or clean_checkpoint or clean_all
        if clean_all:
            _clean_up_all(con, thread_id)

        user_input = await _build_input(con, task_file, paper_file, thread_id)
        project_id, workdir = _init_project(con, thread_id)
        if reset_checkpoint and not clean_all:
            _clean_project_workflows(con, thread_id, project_id)
        async with AsyncRedisSaver.from_conn_string(
            get_settings().persistent.cache.LLM_CACHE.unicode_string()
        ) as checkpoint:
            checkpoint.serde = JsonPlusRedisSerializer(
                allowed_msgpack_modules=[
                    ('app.models.tools', 'FileMount'),
                    ('app.models.tools', 'ImageConfig'),
                    ('app.models.tools', 'ParamDefine'),
                    ('llm.states', 'BuildEdge'),
                    ('llm.states', 'BuildNode'),
                    ('llm.states', 'WorkflowDraft'),
                    ('llm.tools.test_run_report', 'TestRunReport'),
                    ('llm.utils.middleware.todo', 'Todo'),
                    ('llm.utils.uid_mapper', 'BidirectionalMapper'),
                ]
            )

            sample_info = await _get_sample_info(user_input, project_id, workdir, checkpoint)
            standard_plan = await _generate_plan(
                [user_input, HumanMessage(content=sample_info)], project_id, workdir, checkpoint
            )

            agent_input = [
                HumanMessage(content=sample_info),
                HumanMessage(content=standard_plan),
                HumanMessage(
                    content='Please complete the task of constructing a work roadmap based on the above information.'
                ),
            ]
            workflow_result = await _generate_workflow(
                agent_input,
                thread_id,
                project_id,
                workdir,
                checkpoint,
                reset_checkpoint=reset_checkpoint,
            )
        _send_run_task(con, thread_id, project_id, workflow_result, '')
    finally:
        con.close()


@click.group()
def cli() -> None:
    """Command for agent evaluate."""
    from app.utils.logger import init_logging

    init_logging()


@cli.command()
@click.argument('task_file', type=click.Path(exists=True))
@click.option('--paper', '-p', 'paper_file', default=None, type=click.Path(exists=True), help='PDF paper file path.')
@click.option('--thread-id', '-t', default=None, help='Checkpoint thread ID.')
@click.option('--db-index', '-d', default=None, type=click.IntRange(min=1), help='Deprecated; ignored.')
@click.option('--drop-old', is_flag=True, default=False, help='Deprecated alias for --clean-checkpoint.')
@click.option(
    '--clean-checkpoint',
    is_flag=True,
    default=False,
    help='Drop the workflow builder checkpoint and remove workflows imported into the evaluate project.',
)
@click.option(
    '--clean-all', is_flag=True, default=False, help='Delete the evaluate project record and local project files.'
)
def wf_cmd(
    task_file: str,
    paper_file: str | None,
    thread_id: str | None,
    db_index: int | None,
    drop_old: bool,  # noqa: FBT001
    clean_checkpoint: bool,  # noqa: FBT001
    clean_all: bool,  # noqa: FBT001
) -> None:
    """Run the workflow design evaluation on TASK_FILE."""
    asyncio.run(
        wf_eval(
            task_file,
            paper_file,
            thread_id,
            drop_old=drop_old,
            clean_checkpoint=clean_checkpoint,
            clean_all=clean_all,
            db_index=db_index,
        )
    )


if __name__ == '__main__':
    cli()
