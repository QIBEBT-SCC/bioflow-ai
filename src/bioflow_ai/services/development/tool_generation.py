import shutil
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, cast

from anyio import Path
from loguru import logger
from sqlalchemy.orm import Session

from bioflow_ai.llm.observability.execution_logging import AgentRunLogger
from bioflow_ai.llm.observability.tracing import configure_langsmith_tracing
from bioflow_ai.services.agents.execution import consume_agent_graph

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.graph.state import CompiledStateGraph

    from bioflow_ai.llm.context import CreateToolContext
    from bioflow_ai.llm.states import CreateToolOutput
    from bioflow_ai.schemas.tool_config import ImageConfig


from bioflow_ai.services.development.database import (
    DB_PATH,
    TOOL_STATUS_COMPLETED,
    TOOL_STATUS_FAILED,
    TOOL_STATUS_INTERRUPTED,
    TOOL_STATUS_PENDING,
    TOOL_STATUS_RUNNING,
    TOOL_TRAJECTORY_AGENTS,
    row_to_dict,
    rows_to_dicts,
)
from bioflow_ai.services.progress import ProgressCallback, emit_progress


async def _notify_tool_creation(
    event_type: str, title: str, *, tool_name: str, index: int, started_at: float, note: str | None = None
) -> None:
    from bioflow_ai.services.notification.events import format_duration, notify

    duration = timedelta(seconds=time.monotonic() - started_at)
    await notify(
        event_type,
        source='develop.create_tool',
        title=title,
        fields={'Tool': f'**{tool_name}**', 'Index': index, 'Duration': format_duration(duration)},
        note=note,
    )


def _prepare_tool_resources(tool_name: str, help_command: str) -> tuple[uuid.UUID, 'ImageConfig', uuid.UUID] | None:
    """获取工具镜像并复用或创建对应的帮助文档."""
    import bioflow_ai.crud.tools.image as image_crud
    import bioflow_ai.crud.tools.tool_doc as doc_crud
    from bioflow_ai.infrastructure.containers import run_command
    from bioflow_ai.infrastructure.database import engine
    from bioflow_ai.models.tools import ToolHelpDoc

    with Session(engine) as session:
        tool_image = image_crud.get_by_name(session, tool_name)
        if tool_image is None:
            logger.error(f'Image not found: {tool_name}')
            return None

        tool_doc = doc_crud.get_by_command(session, help_command, str(tool_image.uid))
        if tool_doc is None:
            _, content = run_command(tool_image.image, [], help_command)
            tool_doc = doc_crud.insert(
                session,
                ToolHelpDoc(help_command=help_command, content=content, image_uid=tool_image.uid),
            )
            session.commit()

        return tool_image.uid, tool_image.image, tool_doc.uid


@asynccontextmanager
async def _tool_agent_session() -> AsyncGenerator[
    tuple['AsyncPostgresSaver', 'CompiledStateGraph[Any, CreateToolContext, Any, CreateToolOutput]'], None
]:
    """初始化 CreateToolAgent 会话."""
    from bioflow_ai.llm.agents.tool_agent import CreateToolAgent
    from bioflow_ai.llm.checkpoint import open_checkpointer

    async with open_checkpointer(setup=True) as checkpoint:
        agent = CreateToolAgent().build(memory=checkpoint)
        yield checkpoint, agent


async def _cleanup_threads(
    con: Any,
    thread_ids: list[str],
    indices: list[int],
    *,
    delete_trajectories: bool = False,
) -> None:
    """清理指定尝试的检查点和测试产物, 并将开发任务重置为待运行状态.

    普通重置保留持久化轨迹; delete_trajectories=true 时同步删除对应轨迹.
    """
    from bioflow_ai.core.config import get_settings
    from bioflow_ai.infrastructure.containers import remove_container
    from bioflow_ai.llm.observability.trajectories import delete_trajectory_run
    from bioflow_ai.llm.utils.session import delete_checkpoint

    if TYPE_CHECKING:
        from langchain_core.runnables import RunnableConfig

    if thread_ids:
        async with _tool_agent_session() as (checkpoint, agent):
            for thread_id in thread_ids:
                config: RunnableConfig = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 500}
                response = await agent.aget_state(config, subgraphs=True)
                try:
                    # ty: ignore[unresolved-attribute]
                    container_id = cast('str', response.tasks[0].state.values.get('container_id'))
                except (IndexError, AttributeError, KeyError, TypeError):
                    container_id = None
                if container_id:
                    try:
                        remove_container(container_id)
                        logger.info('Removed the leftover container')
                    except Exception as e:
                        logger.warning(e)
                await delete_checkpoint(checkpoint, thread_id)
                logger.info(f'Deleted agent checkpoint: {thread_id}')

        for thread_id in thread_ids:
            output_path = await Path(get_settings().persistent.storage.STORAGE_PATH).absolute() / 'testrun' / thread_id
            if await output_path.exists():
                shutil.rmtree(output_path)
                logger.info(f'Removed run directory: {output_path}')

            if delete_trajectories:
                for agent_name in TOOL_TRAJECTORY_AGENTS:
                    run_id = f'{agent_name}:{thread_id}'
                    if delete_trajectory_run(agent_name, run_id):
                        logger.info(f'Deleted agent trajectory: agent={agent_name}, thread_id={thread_id}')

    if indices:
        idx_placeholders = ','.join(['?' for _ in indices])
        con.execute(
            f"""UPDATE tool_functions
                SET status = ?,
                    thread_id = NULL,
                    tool_uid = NULL,
                    started_at = NULL,
                    finished_at = NULL,
                    last_error = NULL,
                    updated_at = current_timestamp
                WHERE index IN ({idx_placeholders})""",  # noqa: S608
            [TOOL_STATUS_PENDING, *indices],
        )


async def init_tools(
    index: int | None = None, *, langsmith_tracing: bool = False, on_event: ProgressCallback | None = None
) -> None:
    """初始化工具节点."""
    import sys

    import duckdb
    from langchain_core.messages import HumanMessage
    from redis.asyncio import Redis

    from bioflow_ai.core.config import get_settings
    from bioflow_ai.infrastructure.database import engine
    from bioflow_ai.llm.context import CreateToolContext
    from bioflow_ai.schemas.notification import NotificationEventType
    from bioflow_ai.services.tools.catalog import mark_tool_as_ai_checked

    if TYPE_CHECKING:
        from langchain_core.runnables import RunnableConfig

    configure_langsmith_tracing('build-tool-node', enabled=langsmith_tracing)

    skip_tools = ['MACS2', 'RSEM', 'lima', 'awk', 'PhyML', 'Snakemake', 'MrBayes']

    con = duckdb.connect(DB_PATH)

    if index is not None:
        todo_rows = rows_to_dicts(con, 'SELECT * FROM tool_functions WHERE index = ?', [index])
        if not todo_rows:
            logger.error(f'No data found for index {index}')
            con.close()
            return
    else:
        placeholders = ','.join(['?' for _ in skip_tools])
        todo_rows = rows_to_dicts(
            con,
            f'SELECT * FROM tool_functions WHERE tool_name NOT IN ({placeholders}) AND status = ? ORDER BY index ASC',
            [*skip_tools, TOOL_STATUS_PENDING],
        )

    async with _tool_agent_session() as (_, agent):
        for row in todo_rows:
            row_index: int = row['index']
            tool_name: str = row['tool_name']
            function: str = row['function_description']
            help_command: str = row['help_command'].strip()
            logger.info(f'Start create tool: ({row_index}){tool_name}')

            resources = _prepare_tool_resources(tool_name, help_command)
            if resources is None:
                return
            image_uid, image, doc_uid = resources

            agent_input = {
                'messages': [
                    HumanMessage(
                        content=f'<Require>\n'
                        f'tool_name: {tool_name}\n'
                        f'function: {function}\n'
                        f'help_command: `{help_command}`\n'
                        f'</Require>'
                    )
                ],
                'image_uid': image_uid,
                'image': image,
                'doc_uid': doc_uid,
            }

            thread_id = f'tool_{row_index}_{uuid.uuid4().hex}'
            config: RunnableConfig = {'configurable': {'thread_id': thread_id}, 'recursion_limit': sys.maxsize}
            context = CreateToolContext(
                user_id=2,
                skip_search=False,
                auto_save=True,
                save_message=True,
                test_run_id=thread_id,
            )
            notify_tool = partial(
                _notify_tool_creation, tool_name=tool_name, index=row_index, started_at=time.monotonic()
            )

            con.execute(
                """UPDATE tool_functions
                   SET status = ?,
                       thread_id = ?,
                       tool_uid = NULL,
                       attempt_count = attempt_count + 1,
                       started_at = current_timestamp,
                       finished_at = NULL,
                       last_error = NULL,
                       updated_at = current_timestamp
                   WHERE index = ?""",
                [TOOL_STATUS_RUNNING, thread_id, row_index],
            )

            try:
                response = await consume_agent_graph(
                    agent,
                    agent_input,
                    config,
                    context=context,
                    execution_log=AgentRunLogger('tool-generator', thread_id, thread_id=thread_id),
                    on_chunk=lambda chunk: emit_progress(on_event, 'chunk', chunk),
                )
            except Exception as error:
                error_message = f'{type(error).__name__}: {error}'
                con.execute(
                    """UPDATE tool_functions
                       SET status = ?,
                           finished_at = current_timestamp,
                           last_error = ?,
                           updated_at = current_timestamp
                       WHERE index = ?""",
                    [TOOL_STATUS_FAILED, error_message, row_index],
                )
                await notify_tool(
                    NotificationEventType.TOOL_CREATE_FAILED, 'tool creation failed', note=f'Error: {error_message}'
                )
                raise

            if response.interrupts:
                con.execute(
                    """UPDATE tool_functions
                       SET status = ?, finished_at = current_timestamp, updated_at = current_timestamp
                       WHERE index = ?""",
                    [TOOL_STATUS_INTERRUPTED, row_index],
                )
                logger.error(f'{thread_id} interrupted')
                emit_progress(on_event, 'interrupt', response.interrupts)
                await notify_tool(
                    NotificationEventType.TOOL_CREATE_INTERRUPTED,
                    'tool creation interrupted',
                    note=f'Interrupt: {response.interrupts[0].value}',
                )
            else:
                tool_uid = response.values['tool_uid']
                async with Redis.from_url(
                    get_settings().persistent.cache.API_CACHE.unicode_string(),
                    decode_responses=True,
                    socket_connect_timeout=1,
                    socket_timeout=1,
                ) as redis:
                    with Session(engine) as session:
                        await mark_tool_as_ai_checked(session, redis, tool_uid)
                con.execute(
                    """UPDATE tool_functions
                       SET status = ?,
                           tool_uid = ?,
                           finished_at = current_timestamp,
                           updated_at = current_timestamp
                       WHERE index = ?""",
                    [TOOL_STATUS_COMPLETED, tool_uid, row_index],
                )
                logger.success(f'{thread_id} done')
                node_name = response.values.get('name')
                await notify_tool(
                    NotificationEventType.TOOL_CREATE_SUCCEEDED,
                    'tool creation succeeded',
                    note=f'Node name: `{node_name}`'
                    if node_name
                    else 'An existing node was found in the database; no new node was created.',
                )

    con.close()


async def create_tool_manually(
    name: str,
    help_command: str,
    function: str,
    thread_id: str,
    *,
    drop_old: bool,
    langsmith_tracing: bool = False,
    on_event: ProgressCallback | None = None,
) -> None:
    """手动模拟工具生成."""
    import sys

    from langchain_core.messages import HumanMessage

    from bioflow_ai.llm.context import CreateToolContext

    if TYPE_CHECKING:
        from langchain_core.runnables import RunnableConfig

    configure_langsmith_tracing('build-tool-node', enabled=langsmith_tracing)

    async with _tool_agent_session() as (checkpoint, agent):
        if drop_old:
            await checkpoint.adelete_thread(thread_id)

        resources = _prepare_tool_resources(name, help_command)
        if resources is None:
            return
        image_uid, image, doc_uid = resources

        agent_input = {
            'messages': [
                HumanMessage(
                    content=f'<Require>\n'
                    f'tool_name: {name}\n'
                    f'function: {function}\n'
                    f'help_command: `{help_command}`\n'
                    f'</Require>'
                )
            ],
            'image_uid': image_uid,
            'image': image,
            'doc_uid': doc_uid,
        }

        config: RunnableConfig = {'configurable': {'thread_id': thread_id}, 'recursion_limit': sys.maxsize}
        context = CreateToolContext(user_id=2, skip_search=True, auto_save=True, test_run_id=thread_id)

        response = await consume_agent_graph(
            agent,
            agent_input,
            config,
            context=context,
            execution_log=AgentRunLogger('tool-generator', thread_id, thread_id=thread_id),
            on_chunk=lambda chunk: emit_progress(on_event, 'chunk', chunk),
        )
        if response.interrupts:
            logger.error(f'{thread_id} interrupted')
            emit_progress(on_event, 'interrupt', response.interrupts)


async def read_interrupt(index: int, *, on_event: ProgressCallback | None = None) -> None:
    """加载指定中断信息."""
    import duckdb

    if TYPE_CHECKING:
        from langchain_core.runnables import RunnableConfig

    con = duckdb.connect(DB_PATH)
    tool_info = row_to_dict(con, 'SELECT * FROM tool_functions WHERE index = ?', [index])
    con.close()

    if tool_info is None:
        logger.error(f'No data found for index {index}')
        return
    if tool_info['status'] != TOOL_STATUS_INTERRUPTED or not tool_info['thread_id']:
        logger.error(f'Index {index} is not a readable interrupted record: status={tool_info["status"]}')
        return

    async with _tool_agent_session() as (_, agent):
        thread_id = tool_info['thread_id']
        config: RunnableConfig = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 500}

        response = await agent.aget_state(config)
        emit_progress(on_event, 'interrupt', response.interrupts)


async def continue_run(
    index: int,
    message: str | None = None,
    *,
    approve: bool = False,
    langsmith_tracing: bool = False,
    on_event: ProgressCallback | None = None,
) -> None:
    """响应中断信息."""
    import duckdb
    from langgraph.types import Command

    from bioflow_ai.llm.context import CreateToolContext

    if TYPE_CHECKING:
        from langchain_core.runnables import RunnableConfig

    configure_langsmith_tracing('build-tool-node', enabled=langsmith_tracing)

    con = duckdb.connect(DB_PATH)
    tool_info = row_to_dict(con, 'SELECT * FROM tool_functions WHERE index = ?', [index])

    if tool_info is None:
        logger.error(f'No data found for index {index}')
        con.close()
        return
    if tool_info['status'] != TOOL_STATUS_INTERRUPTED or not tool_info['thread_id']:
        logger.error(f'Index {index} is not a resumable interrupted record: status={tool_info["status"]}')
        con.close()
        return

    con.execute(
        """UPDATE tool_functions
           SET status = ?, finished_at = NULL, last_error = NULL, updated_at = current_timestamp
           WHERE index = ?""",
        [TOOL_STATUS_RUNNING, index],
    )

    async with _tool_agent_session() as (_, agent):
        thread_id = tool_info['thread_id']
        config: RunnableConfig = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 500}
        context = CreateToolContext(
            user_id=2,
            skip_search=True,
            auto_save=True,
            save_message=True,
            test_run_id=thread_id,
        )

        if approve:
            input_message = Command(resume=True)
        elif message is not None:
            input_message = Command(resume=message)
        else:
            input_message = None
        try:
            response = await consume_agent_graph(
                agent,
                input_message,
                config,
                context=context,
                execution_log=AgentRunLogger('tool-generator', thread_id, thread_id=thread_id),
                on_chunk=lambda chunk: emit_progress(on_event, 'chunk', chunk),
                resumed=True,
            )
        except Exception as error:
            con.execute(
                """UPDATE tool_functions
                   SET status = ?,
                       finished_at = current_timestamp,
                       last_error = ?,
                       updated_at = current_timestamp
                   WHERE index = ?""",
                [TOOL_STATUS_FAILED, f'{type(error).__name__}: {error}', index],
            )
            raise

        if response.interrupts:
            con.execute(
                """UPDATE tool_functions
                   SET status = ?, finished_at = current_timestamp, updated_at = current_timestamp
                   WHERE index = ?""",
                [TOOL_STATUS_INTERRUPTED, index],
            )
            emit_progress(on_event, 'interrupt', response.interrupts)
        else:
            con.execute(
                """UPDATE tool_functions
                   SET status = ?,
                       tool_uid = ?,
                       finished_at = current_timestamp,
                       updated_at = current_timestamp
                   WHERE index = ?""",
                [TOOL_STATUS_COMPLETED, response.values['tool_uid'], index],
            )

    con.close()


async def delete_prebuild_tool(indices: list[int]) -> None:
    """删除现有节点."""
    import duckdb

    import bioflow_ai.crud.tools.tool as tool_crud
    from bioflow_ai.infrastructure.database import engine
    from bioflow_ai.services.tools.cache import invalidate_tool_projections_sync
    from bioflow_ai.services.tools.index import remove_tool_index

    con = duckdb.connect(DB_PATH)
    thread_ids_to_delete: list[str] = []
    valid_indices: list[int] = []

    for index in indices:
        tool_info = row_to_dict(con, 'SELECT * FROM tool_functions WHERE index = ?', [index])
        if tool_info is None:
            logger.warning(f'No data found for index {index}; skipping')
            continue
        logger.info(f'Data to delete (index={index}): {tool_info}')
        valid_indices.append(index)

        if tool_uid := tool_info.get('tool_uid'):
            with Session(engine) as session:
                tool_crud.delete(session, tool_uid)
                session.commit()
            invalidate_tool_projections_sync(tool_uid)
            logger.info(f'Deleted a database record (index={index})')
            await remove_tool_index(tool_uid)

        if thread_id := tool_info.get('thread_id'):
            thread_ids_to_delete.append(thread_id)

    await _cleanup_threads(con, thread_ids_to_delete, valid_indices, delete_trajectories=True)
    con.close()
    logger.info(f'Successfully deleted {len(valid_indices)} tool records')
