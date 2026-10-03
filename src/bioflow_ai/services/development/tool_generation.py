import shutil
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

from anyio import Path
from loguru import logger
from sqlalchemy.orm import Session

from bioflow_ai.llm.observability.tracing import configure_langsmith_tracing

if TYPE_CHECKING:
    from langgraph.checkpoint.redis import AsyncRedisSaver
    from langgraph.graph.state import CompiledStateGraph

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


def _format_notification_duration(duration_ms: int) -> str:
    seconds = max(0, round(duration_ms / 1000))
    minutes, remainder = divmod(seconds, 60)
    if minutes:
        return f'{minutes} min {remainder} sec'
    return f'{remainder} sec'


async def _publish_tool_creation_event(event_type: str, message: str) -> None:
    """Publish a tool-creation event without changing the underlying task outcome."""
    try:
        from bioflow_ai.services.notification.events import publish_notification

        await publish_notification(
            event_type,
            source='develop.create_tool',
            message=message,
        )
    except Exception:
        logger.exception('The tool creation status was updated, but the notification event could not be published.')


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
async def _tool_agent_session(
    *, setup: bool = False
) -> AsyncGenerator[tuple['AsyncRedisSaver', 'CompiledStateGraph'], None]:
    """初始化 CreateToolAgent 会话."""
    from langgraph.checkpoint.redis import AsyncRedisSaver
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

    from bioflow_ai.core.config import get_settings
    from bioflow_ai.llm.agents.tool_agent import CreateToolAgent

    async with AsyncRedisSaver.from_conn_string(get_settings().develop.CHECKPOINT_URL.unicode_string()) as checkpoint:
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
        if setup:
            await checkpoint.setup()
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
                    # pyrefly: ignore [missing-attribute]
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

    from bioflow_ai.llm.context import CreateToolContext
    from bioflow_ai.schemas.notification import NotificationEventType

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
            f'SELECT * FROM tool_functions WHERE tool_name NOT IN ({placeholders}) AND status = ?',  # noqa: S608
            [*skip_tools, TOOL_STATUS_PENDING],
        )

    completed_count = 0

    async with _tool_agent_session(setup=True) as (_, agent):
        for row in todo_rows:
            row_index: int = row['index']
            tool_name: str = row['tool_name']
            function: str = row['function_description']
            help_command: str = row['help_command'].strip()

            resources = _prepare_tool_resources(tool_name, help_command)
            if resources is None:
                return
            image_uid, image, doc_uid = resources

            agent_input = {
                'messages': [
                    HumanMessage(
                        content=f'<Require>\n'
                        f'tool_name: {tool_name}\n'
                        f'function: {function}\nhelp_command: `{help_command}`\n'
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
            started_at_monotonic = time.monotonic()

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
                # pyrefly: ignore [no-matching-overload]
                async for chunk in agent.astream(
                    agent_input,
                    config,
                    context=context,
                    stream_mode=['custom', 'updates'],
                    version='v2',
                    subgraphs=True,
                ):
                    emit_progress(on_event, 'chunk', chunk)

                response = await agent.aget_state(config)
            except Exception as error:
                error_message = f'{type(error).__name__}: {error}'
                duration_ms = round((time.monotonic() - started_at_monotonic) * 1000)
                con.execute(
                    """UPDATE tool_functions
                       SET status = ?,
                           finished_at = current_timestamp,
                           last_error = ?,
                           updated_at = current_timestamp
                       WHERE index = ?""",
                    [TOOL_STATUS_FAILED, error_message, row_index],
                )
                await _publish_tool_creation_event(
                    NotificationEventType.TOOL_CREATE_FAILED,
                    '## BioFlow AI tool creation failed\n'
                    f'- Tool: **{tool_name}**\n'
                    f'- Index: {row_index}\n'
                    f'- Duration: {_format_notification_duration(duration_ms)}\n'
                    f'> Error: `{type(error).__name__}` {str(error)[:1000]}',
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
                duration_ms = round((time.monotonic() - started_at_monotonic) * 1000)
                await _publish_tool_creation_event(
                    NotificationEventType.TOOL_CREATE_INTERRUPTED,
                    '## BioFlow AI tool creation interrupted\n'
                    f'- Tool: **{tool_name}**\n'
                    f'- Index: {row_index}\n'
                    f'- Duration: {_format_notification_duration(duration_ms)}\n'
                    f'> Interrupt: {response.interrupts[0].value}',
                )
            else:
                tool_uid = response.values['tool_uid']
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
                completed_count += 1
                duration_ms = round((time.monotonic() - started_at_monotonic) * 1000)
                if node_name := response.values.get('name'):
                    await _publish_tool_creation_event(
                        NotificationEventType.TOOL_CREATE_SUCCEEDED,
                        '## BioFlow AI tool creation succeeded\n'
                        f'- Tool: **{tool_name}**\n'
                        f'- Index: {row_index}\n'
                        f'- Node name: `{node_name}`\n'
                        f'- Duration: {_format_notification_duration(duration_ms)}',
                    )
                else:
                    await _publish_tool_creation_event(
                        NotificationEventType.TOOL_CREATE_SUCCEEDED,
                        '## BioFlow AI tool creation succeeded\n'
                        f'- Tool: **{tool_name}**\n'
                        f'- Index: {row_index}\n'
                        f'- Duration: {_format_notification_duration(duration_ms)}\n'
                        'An existing node was found in the database; no new node was created.',
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

        # pyrefly: ignore [no-matching-overload]
        async for chunk in agent.astream(
            agent_input,
            config,
            context=context,
            stream_mode=['custom', 'updates'],
            version='v2',
            subgraphs=True,
        ):
            emit_progress(on_event, 'chunk', chunk)

        response = await agent.aget_state(config)
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
            # pyrefly: ignore [no-matching-overload]
            async for chunk in agent.astream(
                input_message,
                config,
                context=context,
                stream_mode=['custom', 'updates'],
                version='v2',
                subgraphs=True,
            ):
                emit_progress(on_event, 'chunk', chunk)

            response = await agent.aget_state(config)
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
    from bioflow_ai.core.config import get_settings
    from bioflow_ai.infrastructure.database import engine
    from bioflow_ai.llm.vectorstores import MilvusVectorStore

    con = duckdb.connect(DB_PATH)
    thread_ids_to_delete: list[str] = []
    valid_indices: list[int] = []
    vectorstore = MilvusVectorStore(
        collection_name='ToolCollection',
        uri=get_settings().knowledge_base.MILVUS_URL,
        hybrid=True,
    )

    async with vectorstore:
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
                    logger.info(f'Deleted a database record (index={index})')
                await vectorstore.adelete(
                    filter_expression='tool_uid == {uid}',
                    filter_params={'uid': tool_uid},
                )

            if thread_id := tool_info.get('thread_id'):
                thread_ids_to_delete.append(thread_id)

    await _cleanup_threads(con, thread_ids_to_delete, valid_indices, delete_trajectories=True)
    con.close()
    logger.info(f'Successfully deleted {len(valid_indices)} tool records')
