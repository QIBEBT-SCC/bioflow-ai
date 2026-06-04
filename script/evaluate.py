import asyncio
import sys
from typing import Any

import click
from _duckdb import DuckDBPyConnection
from langchain_core.messages import HumanMessage
from loguru import logger

DB_PATH = 'presets/build/evaluate.db'


def _check_db() -> DuckDBPyConnection:
    import duckdb

    con = duckdb.connect(DB_PATH)

    con.execute("""
        CREATE TABLE IF NOT EXISTS evaluate_results (
            thread_id    VARCHAR PRIMARY KEY,
            task_prompt  VARCHAR NOT NULL,
            paper_file   VARCHAR,
            project_id   INTEGER,
            sample_uid   VARCHAR,
            workflow_uid VARCHAR,
            output_path  VARCHAR
        )
    """)

    return con


def _clean_up(con: DuckDBPyConnection, thread_id: str) -> None:
    from sqlmodel import Session

    import app.crud.project as proj_crud
    from app.core.session import engine

    cursor = con.execute(
        'SELECT * FROM evaluate_results WHERE thread_id = $thread_id',
        {'thread_id': thread_id},
    )
    raw = cursor.fetchone()
    row = dict(zip([desc[0] for desc in cursor.description], raw, strict=False)) if raw else None

    if row and row['project_id']:
        with Session(engine) as session:
            proj_crud.delete(session, row['project_id'])

    con.execute('DELETE FROM evaluate_results WHERE thread_id = $thread_id', {'thread_id': thread_id})


def _init_project(con: DuckDBPyConnection, thread_id: str) -> int:
    cursor = con.execute(
        'SELECT project_id FROM evaluate_results WHERE thread_id = $thread_id',
        {'thread_id': thread_id},
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
        return row[0]

    from sqlmodel import Session

    import app.crud.project as proj_crud
    from app.core.session import engine
    from app.models.project import Project

    with Session(engine) as session:
        task_project = Project(name=f'wf_{thread_id}', owner_id=2, public=False)
        task_project = proj_crud.insert(session, task_project)

    con.execute(
        'UPDATE evaluate_results SET project_id = $project_id WHERE thread_id = $thread_id',
        {'thread_id': thread_id, 'project_id': task_project.id},
    )

    return task_project.id


async def _build_input(con: DuckDBPyConnection, task_file: str, paper_file: str | None, thread_id: str) -> HumanMessage:
    import base64
    from pathlib import Path

    from llm.utils.multimodal import filter_multimedia_message

    task_data = {'thread_id': thread_id}
    task_path = Path(task_file)
    if not task_path.exists():
        logger.error('Task file not found!')
        sys.exit(0)
    task_data['task_prompt'] = task_path.read_text(encoding='utf-8')

    if paper_file:
        paper_path = Path(paper_file)
        if not paper_path.exists():
            logger.error('Paper file not found!')
            sys.exit(0)
        task_data['paper_file'] = paper_file

        base64_encoded = base64.b64encode(paper_path.read_bytes()).decode('utf-8')
        init_human_message = HumanMessage(
            content=[
                {
                    'type': 'text',
                    'text': task_path.read_text(encoding='utf-8'),
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
                    'text': task_path.read_text(encoding='utf-8'),
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
        VALUES ($thread_id, $task_prompt, $paper_file)
        ON CONFLICT (thread_id) DO UPDATE SET
            task_prompt = EXCLUDED.task_prompt,
            paper_file  = EXCLUDED.paper_file
        """,
        task_data,
    )

    return human_input


async def _generate_workflow(human_input: Any, thread_id: str, project_id: int, *, drop_old: bool = False) -> str:
    import rich
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.redis import AsyncRedisSaver
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
    from rich.console import Console

    from app.core.config import get_settings
    from llm.agents.workflow_agent import WorkflowDesignAgent
    from llm.states import DesignContext
    from llm.utils.render import print_stream_chunk
    from llm.utils.session import delete_checkpoint

    console = Console()

    async with AsyncRedisSaver.from_conn_string(
        get_settings().persistent.cache.LLM_CACHE.unicode_string()
    ) as checkpoint:
        checkpoint.serde = JsonPlusRedisSerializer(
            allowed_msgpack_modules=[
                ('llm.states', 'BuildEdge'),
                ('llm.states', 'BuildNode'),
                ('llm.utils.middleware.todo', 'Todo'),
                ('llm.utils.uid_mapper', 'BidirectionalMapper'),
                ('app.models.tools', 'FileMount'),
                ('app.models.tools', 'ImageConfig'),
                ('app.models.tools', 'ParamDefine'),
            ]
        )
        if drop_old:
            await delete_checkpoint(checkpoint, thread_id)

        agent = WorkflowDesignAgent(checkpoint).build()
        config = RunnableConfig(configurable={'thread_id': thread_id}, recursion_limit=sys.maxsize)
        agent_input = {'messages': [human_input]}

        async for chunk in agent.astream(
            agent_input,
            config,
            context=DesignContext(user_id=2, project_id=project_id, auto_approve=True),
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

        workflow_uid = state.values.get('workflow_uids', [])[0]
        logger.success(f'工作流已生成: {workflow_uid}')
        return workflow_uid


def _send_run_task(
    con: DuckDBPyConnection, thread_id: str, project_id: int, workflow_uid: str, output_dir: str
) -> None:
    import uuid

    from sqlmodel import Session

    import app.crud.project_workflow as proj_wf_crud
    from app.core.session import engine
    from app.models.user import User

    con.execute(
        'UPDATE evaluate_results SET output_path = $output_path WHERE thread_id = $thread_id',
        {'thread_id': thread_id, 'output_path': output_dir},
    )

    with Session(engine) as session:
        proj_wf_crud.add_workflow_to_project(session, project_id, uuid.UUID(workflow_uid))


async def wf_eval(task_file: str, paper_file: str | None, thread_id: str, *, drop_old: bool) -> None:
    """运行bioflow ai 测试."""
    import os

    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_PROJECT'] = 'bioflow-ai-eval'

    con = _check_db()
    if drop_old:
        _clean_up(con, thread_id)

    agent_input = await _build_input(con, task_file, paper_file, thread_id)
    project_id = _init_project(con, thread_id)
    workflow_uid = await _generate_workflow(agent_input, thread_id, project_id, drop_old=drop_old)
    _send_run_task(con, thread_id, project_id, workflow_uid, '')
    con.close()


@click.group()
def cli() -> None:
    """Command for agent evaluate."""
    from app.utils.logger import init_logging

    init_logging()


@cli.command()
@click.argument('task_file', type=click.Path(exists=True))
@click.option('--paper', '-p', 'paper_file', default=None, type=click.Path(exists=True), help='PDF paper file path.')
@click.option('--thread-id', '-t', help='Checkpoint thread ID.')
@click.option('--drop-old', is_flag=True, default=False, help='Drop existing checkpoint before running.')
def wf_cmd(task_file: str, paper_file: str | None, thread_id: str, drop_old: bool) -> None:  # noqa: FBT001
    """Run the workflow design evaluation on TASK_FILE."""
    asyncio.run(wf_eval(task_file, paper_file, thread_id, drop_old=drop_old))


if __name__ == '__main__':
    cli()
