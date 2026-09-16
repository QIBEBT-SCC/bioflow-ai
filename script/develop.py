import asyncio
import shutil
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path as FilePath
from typing import TYPE_CHECKING, Any, cast

import click
import orjson
from anyio import Path
from loguru import logger
from sqlalchemy.orm import Session

from llm.utils.lc.tracing import configure_langsmith_tracing

if TYPE_CHECKING:
    from langgraph.checkpoint.redis import AsyncRedisSaver
    from langgraph.graph.state import CompiledStateGraph

    from app.schemas.tool_config import ImageConfig


DB_PATH = 'presets/build/develop.db'
TRAJECTORY_DATASET_DIRECTORY = 'agent_datasets/trajectories'
TRAJECTORY_INSERT_BATCH_SIZE = 500
TOOL_STATUS_PENDING = 'PENDING'
TOOL_STATUS_RUNNING = 'RUNNING'
TOOL_STATUS_INTERRUPTED = 'INTERRUPTED'
TOOL_STATUS_FAILED = 'FAILED'
TOOL_STATUS_COMPLETED = 'COMPLETED'
TOOL_INCOMPLETE_STATUSES = (TOOL_STATUS_RUNNING, TOOL_STATUS_INTERRUPTED, TOOL_STATUS_FAILED)
TOOL_TRAJECTORY_AGENTS = ('test_run', 'tool_generator')


def _format_notification_duration(duration_ms: int) -> str:
    seconds = max(0, round(duration_ms / 1000))
    minutes, remainder = divmod(seconds, 60)
    if minutes:
        return f'{minutes} min {remainder} sec'
    return f'{remainder} sec'


async def _publish_tool_creation_event(event_type: str, message: str) -> None:
    """Publish a tool-creation event without changing the underlying task outcome."""
    try:
        from app.services.notification.events import publish_notification

        await publish_notification(
            event_type,
            source='develop.create_tool',
            message=message,
        )
    except Exception:
        logger.exception('The tool creation status was updated, but the notification event could not be published.')


def _default_trajectory_dataset_path() -> str:
    """Return the configured persistent destination for compacted trajectories."""
    from app.core.config import get_settings

    storage_root = FilePath(get_settings().persistent.storage.STORAGE_PATH).expanduser().resolve()
    return str(storage_root / TRAJECTORY_DATASET_DIRECTORY)


def _load_trajectory_event(event_file: FilePath) -> dict[str, Any]:
    event = orjson.loads(event_file.read_bytes())
    if not isinstance(event, dict):
        msg = 'Top-level JSON value must be an object.'
        raise TypeError(msg)
    return event


def _raise_click_exception(message: str) -> None:
    raise click.ClickException(message)


def _trajectory_event_row(event: dict[str, Any], source_file: FilePath) -> tuple[Any, ...]:
    """Convert one trajectory event to the stable offline dataset schema."""
    required_fields = ('schema_version', 'event_id', 'agent', 'run_id', 'event_type', 'recorded_at_ns')
    missing_fields = [field for field in required_fields if event.get(field) is None]
    if missing_fields:
        msg = f'Missing required field(s): {", ".join(missing_fields)}'
        raise ValueError(msg)

    return (
        event['schema_version'],
        event['event_id'],
        event['agent'],
        event['run_id'],
        event.get('thread_id'),
        event.get('user_id'),
        event['event_type'],
        event.get('started_at'),
        event.get('ended_at'),
        event.get('recorded_at'),
        event['recorded_at_ns'],
        event.get('process_id'),
        event.get('duration_ms'),
        orjson.dumps(event.get('payload')).decode(),
        orjson.dumps(event.get('error')).decode() if event.get('error') is not None else None,
        str(source_file),
    )


def compact_trajectories(
    input_dir: str,
    output_dir: str,
    agents: tuple[str, ...],
    *,
    skip_invalid: bool,
    overwrite: bool,
) -> None:
    """Validate immutable trajectory JSON events and compact them into Parquet datasets."""
    import duckdb

    source_root = FilePath(input_dir).resolve()
    destination = FilePath(output_dir).resolve()
    if not source_root.is_dir():
        msg = f'Trajectory directory does not exist: {source_root}'
        raise click.ClickException(msg)
    if destination.exists() and not overwrite:
        msg = f'Output directory already exists: {destination}. Use --overwrite to replace it.'
        raise click.ClickException(msg)

    event_files = sorted(source_root.glob('agent=*/run=*/events/*.json'))
    if not event_files:
        msg = f'No trajectory events found: {source_root}'
        raise click.ClickException(msg)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_output = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.tmp')
    temp_output.mkdir()
    con = duckdb.connect()
    invalid_files: list[dict[str, str]] = []
    selected_count = 0

    try:
        con.execute("""
            CREATE TABLE trajectory_events (
                schema_version INTEGER,
                event_id VARCHAR PRIMARY KEY,
                agent VARCHAR NOT NULL,
                run_id VARCHAR NOT NULL,
                thread_id VARCHAR,
                user_id BIGINT,
                event_type VARCHAR NOT NULL,
                started_at TIMESTAMPTZ,
                ended_at TIMESTAMPTZ,
                recorded_at TIMESTAMPTZ,
                recorded_at_ns UBIGINT NOT NULL,
                process_id BIGINT,
                duration_ms DOUBLE,
                payload JSON,
                error JSON,
                source_file VARCHAR NOT NULL
            )
        """)
        insert_sql = 'INSERT INTO trajectory_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
        batch: list[tuple[Any, ...]] = []
        for event_file in event_files:
            try:
                event = _load_trajectory_event(event_file)
                if agents and event.get('agent') not in agents:
                    continue
                batch.append(_trajectory_event_row(event, event_file.relative_to(source_root)))
                selected_count += 1
                if len(batch) >= TRAJECTORY_INSERT_BATCH_SIZE:
                    con.executemany(insert_sql, batch)
                    batch.clear()
            except Exception as error:
                invalid_files.append({'file': str(event_file), 'error': str(error)})
                if not skip_invalid:
                    msg = f'Invalid trajectory file {event_file}: {error}'
                    raise click.ClickException(msg) from error
        if batch:
            con.executemany(insert_sql, batch)

        if selected_count == 0:
            selected_agents = ', '.join(agents) if agents else '(all)'
            msg = f'No events match the agent filter: {selected_agents}'
            _raise_click_exception(msg)

        events_path = temp_output / 'events.parquet'
        runs_path = temp_output / 'runs.parquet'
        con.execute(
            """
            COPY (
                SELECT *
                FROM trajectory_events
                ORDER BY agent, run_id, recorded_at_ns, event_id
            ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
            """,
            [str(events_path)],
        )
        con.execute(
            """
            COPY (
                SELECT
                    agent,
                    run_id,
                    any_value(thread_id) AS thread_id,
                    any_value(user_id) AS user_id,
                    min(recorded_at) AS started_at,
                    max(recorded_at) AS ended_at,
                    CASE
                        WHEN arg_max(event_type, recorded_at_ns) = 'agent_end' THEN 'completed'
                        WHEN count(*) FILTER (WHERE error IS NOT NULL) > 0 THEN 'failed'
                        ELSE 'incomplete'
                    END AS status,
                    count(*) AS event_count,
                    count(*) FILTER (WHERE event_type = 'agent_start') AS agent_start_count,
                    count(*) FILTER (WHERE event_type = 'model_call') AS model_call_count,
                    count(*) FILTER (WHERE event_type = 'tool_call') AS tool_call_count,
                    count(*) FILTER (WHERE event_type = 'agent_end') AS agent_end_count,
                    count(*) FILTER (WHERE error IS NOT NULL) AS error_count,
                    min(schema_version) AS min_schema_version,
                    max(schema_version) AS max_schema_version
                FROM trajectory_events
                GROUP BY agent, run_id
                ORDER BY agent, run_id
            ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
            """,
            [str(runs_path)],
        )

        run_count_row = con.execute('SELECT count(DISTINCT run_id) FROM trajectory_events').fetchone()
        run_count = cast('tuple[int]', run_count_row)[0]
        agent_counts = {
            row[0]: row[1]
            for row in con.execute(
                'SELECT agent, count(DISTINCT run_id) FROM trajectory_events GROUP BY agent ORDER BY agent'
            ).fetchall()
        }
        metadata = {
            'schema_version': 1,
            'generated_at': datetime.now(UTC).isoformat(),
            'source_root': str(source_root),
            'agent_filter': list(agents),
            'event_count': selected_count,
            'run_count': run_count,
            'run_count_by_agent': agent_counts,
            'invalid_file_count': len(invalid_files),
            'invalid_files': invalid_files,
        }
        (temp_output / 'metadata.json').write_bytes(
            orjson.dumps(metadata, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE)
        )
    except Exception:
        shutil.rmtree(temp_output, ignore_errors=True)
        raise
    finally:
        con.close()

    backup = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.backup')
    try:
        if destination.exists():
            destination.replace(backup)
        temp_output.replace(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)

    logger.success(
        f'Trajectory compaction complete: {selected_count} events, {run_count} runs, '
        f'{len(invalid_files)} invalid files; output directory: {destination}'
    )


def _row_to_dict(con: Any, query: str, params: list | None = None) -> dict[str, Any] | None:
    cur = con.execute(query, params or [])
    row = cur.fetchone()
    if row is None:
        return None
    return dict(zip([d[0] for d in cur.description], row, strict=False))


def _rows_to_dicts(con: Any, query: str, params: list | None = None) -> list[dict[str, Any]]:
    cur = con.execute(query, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


def _prepare_tool_resources(tool_name: str, help_command: str) -> tuple[uuid.UUID, 'ImageConfig', uuid.UUID] | None:
    """获取工具镜像并复用或创建对应的帮助文档."""
    import app.crud.tools.image as image_crud
    import app.crud.tools.tool_doc as doc_crud
    from app.core.session import engine
    from app.models.tools import ToolHelpDoc
    from app.utils.container import run_command

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

    from app.core.config import get_settings
    from llm.agents.tool_agent import CreateToolAgent

    async with AsyncRedisSaver.from_conn_string(get_settings().develop.CHECKPOINT_URL.unicode_string()) as checkpoint:
        checkpoint.serde = JsonPlusRedisSerializer(
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
    from app.core.config import get_settings
    from app.utils.container import remove_container
    from llm.utils.session import delete_checkpoint
    from llm.utils.trajectory_store import delete_trajectory_run

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


def search_tools(tool_name: str) -> None:
    """搜索数据库中指定名称的工具原始数据."""
    import duckdb
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    con = duckdb.connect(DB_PATH)
    result = con.execute('SELECT * FROM tool_functions WHERE tool_name LIKE ?', [f'%{tool_name}%']).pl()

    console = Console()
    for row in result.iter_rows(named=True):
        content = Text()
        content.append('Tool: ', style='bold cyan')
        content.append(f'{row["tool_name"]}\n', style='magenta')
        content.append('Status: ', style='bold cyan')
        status = row['status']
        status_style = (
            'green' if status == TOOL_STATUS_COMPLETED else 'yellow' if status == TOOL_STATUS_PENDING else 'red'
        )
        content.append(f'{status}\n', style=status_style)
        content.append('Tool UID: ', style='bold cyan')
        content.append(f'{row["tool_uid"]}\n', style='white')
        content.append('Description:\n', style='bold cyan')
        content.append(row['function_description'], style='white')

        console.print(
            Panel(
                content,
                title=f'[yellow]#{row["index"]} {row["thread_id"]}[/yellow]',
                border_style='blue',
                expand=True,
            )
        )

    con.close()


async def init_tools(index: int | None = None, *, langsmith_tracing: bool = False) -> None:
    """初始化工具节点."""
    import sys

    import duckdb
    import rich
    from langchain_core.messages import HumanMessage
    from rich.console import Console

    from app.schemas.notification import NotificationEventType
    from llm.context import CreateToolContext
    from llm.utils.render import print_stream_chunk

    if TYPE_CHECKING:
        from langchain_core.runnables import RunnableConfig

    configure_langsmith_tracing('build-tool-node', enabled=langsmith_tracing)

    skip_tools = ['MACS2', 'RSEM', 'lima', 'awk', 'PhyML', 'Snakemake', 'MrBayes']

    con = duckdb.connect(DB_PATH)

    if index is not None:
        todo_rows = _rows_to_dicts(con, 'SELECT * FROM tool_functions WHERE index = ?', [index])
        if not todo_rows:
            logger.error(f'No data found for index {index}')
            con.close()
            return
    else:
        placeholders = ','.join(['?' for _ in skip_tools])
        todo_rows = _rows_to_dicts(
            con,
            f'SELECT * FROM tool_functions WHERE tool_name NOT IN ({placeholders}) AND status = ?',  # noqa: S608
            [*skip_tools, TOOL_STATUS_PENDING],
        )

    console = Console()
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
                    print_stream_chunk(console, chunk)

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
                rich.print(response.interrupts)
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
) -> None:
    """手动模拟工具生成."""
    import sys

    import rich
    from langchain_core.messages import HumanMessage
    from rich.console import Console

    from llm.context import CreateToolContext
    from llm.utils.render import print_stream_chunk

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

        console = Console()
        # pyrefly: ignore [no-matching-overload]
        async for chunk in agent.astream(
            agent_input,
            config,
            context=context,
            stream_mode=['custom', 'updates'],
            version='v2',
            subgraphs=True,
        ):
            print_stream_chunk(console, chunk)

        response = await agent.aget_state(config)
        if response.interrupts:
            logger.error(f'{thread_id} interrupted')
            rich.print(response.interrupts)


async def read_interrupt(index: int) -> None:
    """加载指定中断信息."""
    import duckdb
    import rich

    if TYPE_CHECKING:
        from langchain_core.runnables import RunnableConfig

    con = duckdb.connect(DB_PATH)
    tool_info = _row_to_dict(con, 'SELECT * FROM tool_functions WHERE index = ?', [index])
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
        rich.print(response.interrupts)


async def continue_run(
    index: int,
    message: str | None = None,
    *,
    approve: bool = False,
    langsmith_tracing: bool = False,
) -> None:
    """响应中断信息."""
    import duckdb
    import rich
    from langgraph.types import Command
    from rich.console import Console

    from llm.context import CreateToolContext
    from llm.utils.render import print_stream_chunk

    if TYPE_CHECKING:
        from langchain_core.runnables import RunnableConfig

    configure_langsmith_tracing('build-tool-node', enabled=langsmith_tracing)

    con = duckdb.connect(DB_PATH)
    tool_info = _row_to_dict(con, 'SELECT * FROM tool_functions WHERE index = ?', [index])

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

    console = Console()

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
                print_stream_chunk(console, chunk)

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
            rich.print(response.interrupts)
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

    import app.crud.tools.tool as tool_crud
    from app.core.config import get_settings
    from app.core.session import engine
    from llm.vectorstores import MilvusVectorStore

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
            tool_info = _row_to_dict(con, 'SELECT * FROM tool_functions WHERE index = ?', [index])
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


def statistic() -> None:
    """统计生成情况."""
    import duckdb
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    con = duckdb.connect(DB_PATH, read_only=True)

    total = (con.execute('SELECT COUNT(*) FROM tool_functions').fetchone() or (0,))[0]
    pending = (
        con.execute('SELECT COUNT(*) FROM tool_functions WHERE status = ?', [TOOL_STATUS_PENDING]).fetchone() or (0,)
    )[0]
    processed = total - pending
    success = (
        con.execute('SELECT COUNT(*) FROM tool_functions WHERE status = ?', [TOOL_STATUS_COMPLETED]).fetchone() or (0,)
    )[0]
    unique_nodes = (
        con.execute('SELECT COUNT(DISTINCT tool_uid) FROM tool_functions WHERE tool_uid IS NOT NULL').fetchone() or (0,)
    )[0]
    failed = (
        con.execute('SELECT COUNT(*) FROM tool_functions WHERE status = ?', [TOOL_STATUS_FAILED]).fetchone() or (0,)
    )[0]
    interrupted = (
        con.execute('SELECT COUNT(*) FROM tool_functions WHERE status = ?', [TOOL_STATUS_INTERRUPTED]).fetchone()
        or (0,)
    )[0]
    running = (
        con.execute('SELECT COUNT(*) FROM tool_functions WHERE status = ?', [TOOL_STATUS_RUNNING]).fetchone() or (0,)
    )[0]

    console = Console()

    stats = Table.grid(padding=(0, 2))
    stats.add_column(style='bold', justify='right')
    stats.add_column()

    pct = (processed / total * 100) if total else 0

    stats.add_row('Total records', f'[cyan]{total:,}[/cyan]')
    stats.add_row('Processed', f'[cyan]{processed:,}[/cyan] [dim]({pct:.1f}%)[/dim]')
    stats.add_row('Succeeded', f'[green]{success:,}[/green]  Nodes generated [green]{unique_nodes:,}[/green]')
    stats.add_row(
        'Incomplete',
        f'Running [cyan]{running:,}[/cyan]  Interrupted [yellow]{interrupted:,}[/yellow]  Failed [red]{failed:,}[/red]',
    )

    console.print(
        Panel(
            stats,
            title='[bold]Run Statistics[/bold]',
            border_style='blue',
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )

    def _render_record_panel(row: dict | None, title: str, border_style: str = 'cyan') -> None:
        if not row:
            console.print(Panel('[dim](No data)[/dim]', title=title, border_style=border_style))
            return

        body = Table.grid(padding=(0, 2))
        body.add_column(style='bold cyan', justify='right', no_wrap=True)
        body.add_column(overflow='fold')

        for key, value in row.items():
            if value is None:
                display = '[dim]None[/dim]'
            elif key == 'status':
                style = (
                    'green' if value == TOOL_STATUS_COMPLETED else 'yellow' if value == TOOL_STATUS_PENDING else 'red'
                )
                display = f'[{style}]{value}[/{style}]'
            else:
                display = str(value)
            body.add_row(f'{key}:', display)

        console.print(
            Panel(
                body,
                title=title,
                border_style=border_style,
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    latest = _row_to_dict(
        con,
        'SELECT * FROM tool_functions WHERE status <> ? ORDER BY started_at DESC NULLS LAST LIMIT 1',
        [TOOL_STATUS_PENDING],
    )
    _render_record_panel(latest, '[bold green]Latest Run Record[/bold green]')

    latest_interrupt = _row_to_dict(
        con,
        'SELECT * FROM tool_functions WHERE status = ? ORDER BY finished_at DESC NULLS LAST LIMIT 1',
        [TOOL_STATUS_INTERRUPTED],
    )
    _render_record_panel(latest_interrupt, '[bold yellow]Latest Interrupted Record[/bold yellow]')

    con.close()


@click.group()
def cli() -> None:
    """Developer tools."""
    from app.utils.logger import init_logging

    init_logging()


@cli.command()
@click.option('--name', '-n', type=str, default=None, help='Tool name to search for.')
def search_cmd(name: str) -> None:
    """Search for tools by name."""
    search_tools(tool_name=name)


@cli.command()
@click.option('--index', type=int, default=None, help='Data index to process (default: all incomplete records).')
@click.option('--name', '-n', help='Tool name; switches to manual creation mode when provided.')
@click.option('--help-command', '-c', help='Command used to retrieve tool help (required in manual mode).')
@click.option('--function', '-f', help='Description of the tool function to generate (required in manual mode).')
@click.option('--thread-id', '-t', help='Checkpoint thread ID (required in manual mode).')
@click.option(
    '--drop-old', is_flag=True, default=False, help='Delete the old checkpoint for this thread before running.'
)
@click.option(
    '--langsmith-tracing', is_flag=True, default=False, help='Enable LangSmith tracing (disabled by default).'
)
def create_tools_cmd(
    index: int | None,
    name: str | None,
    help_command: str | None,
    function: str | None,
    thread_id: str | None,
    *,
    drop_old: bool,
    langsmith_tracing: bool,
) -> None:
    """Create tools automatically, or create one manually with --name."""
    manual_values = {
        '--name': name,
        '--help-command': help_command,
        '--function': function,
        '--thread-id': thread_id,
    }
    if any(value is not None for value in manual_values.values()):
        missing = [option for option, value in manual_values.items() if value is None]
        if missing:
            message = f'Missing options for manual creation mode: {", ".join(missing)}'
            raise click.UsageError(message)
        if index is not None:
            message = '--index cannot be used with manual creation options'
            raise click.UsageError(message)
        # The completeness check above narrows these values at runtime.
        asyncio.run(
            create_tool_manually(
                cast('str', name),
                cast('str', help_command),
                cast('str', function),
                cast('str', thread_id),
                drop_old=drop_old,
                langsmith_tracing=langsmith_tracing,
            )
        )
        return

    if drop_old:
        message = '--drop-old can only be used in manual creation mode'
        raise click.UsageError(message)
    asyncio.run(init_tools(index, langsmith_tracing=langsmith_tracing))


@cli.command()
@click.argument('index', type=int)
def read_interrupt_cmd(index: int) -> None:
    """Read tool interruption details."""
    asyncio.run(read_interrupt(index))


@cli.command()
@click.argument('index', type=int)
@click.option('--message', '-m', type=str, default=None, help='Message to send when resuming the run.')
@click.option('--approve', is_flag=True, default=False, help='Approve the interrupted action.')
@click.option(
    '--langsmith-tracing', is_flag=True, default=False, help='Enable LangSmith tracing (disabled by default).'
)
def continue_run_cmd(index: int, message: str | None, *, approve: bool, langsmith_tracing: bool) -> None:
    """Resume the tool creation task at the specified index."""
    asyncio.run(continue_run(index, message, approve=approve, langsmith_tracing=langsmith_tracing))


@cli.command()
@click.argument('indices', type=int, nargs=-1, required=True)
def delete_prebuild_tool_cmd(indices: tuple[int, ...]) -> None:
    """Delete prebuilt tools by index (multiple indices are supported, e.g. 61 62 63 65)."""
    asyncio.run(delete_prebuild_tool(list(indices)))


@cli.command()
@click.option('--upload', is_flag=True, default=False, help='Upload the exported data to Hugging Face.')
@click.option(
    '--db',
    type=click.Choice(['images', 'docs', 'tools']),
    multiple=True,
    default=['images', 'docs', 'tools'],
    show_default=True,
    help='Select one or more datasets to export.',
)
def export(*, upload: bool, db: tuple[str, ...]) -> None:
    """Export database data."""
    from app.utils.export.database import export_docs, export_images, export_tools

    db_map = {
        'images': export_images,
        'docs': export_docs,
        'tools': export_tools,
    }
    for name in db:
        logger.info(f'Exporting {name}...')
        result = db_map[name](upload=upload)
        logger.success('Exported {} records to {}', result.record_count, result.output)
        if result.remote_path is not None:
            logger.success('Uploaded {} to {}', result.name, result.remote_path)


@cli.command(name='export-mcp')
@click.option('--upload', is_flag=True, default=False, help='Upload tools.duckdb to the Hugging Face data bucket.')
@click.option(
    '--output',
    required=True,
    type=click.Path(dir_okay=False, path_type=FilePath),
    help='Destination DuckDB file for the standalone MCP server.',
)
def export_mcp_cmd(output: FilePath, *, upload: bool) -> None:
    """Export the local MCP tool knowledge database."""
    from app.utils.export.mcp import export_mcp_database

    try:
        result = export_mcp_database(output, upload=upload)
    except Exception as error:
        raise click.ClickException(str(error)) from error

    logger.success('Exported MCP database to {} ({} tools)', result.output, result.tool_count)
    if result.remote_path is not None:
        logger.success('Uploaded MCP database to {}', result.remote_path)


@cli.command()
def statistic_cmd() -> None:
    """Show node generation statistics."""
    statistic()


@cli.command(name='export-node-stats')
@click.option(
    '--db-path',
    type=click.Path(dir_okay=False, path_type=FilePath),
    default=DB_PATH,
    show_default=True,
    help='DuckDB file containing tool_functions run records.',
)
@click.option(
    '--input-dir',
    type=click.Path(file_okay=False, path_type=FilePath),
    default=None,
    help='Saved agent trajectories (default: persistent.storage.storage_path/agent_trajectories).',
)
@click.option(
    '--output',
    type=click.Path(dir_okay=False, path_type=FilePath),
    default=FilePath('presets/build/node_generation_stats.csv'),
    show_default=True,
    help='Destination CSV file.',
)
def export_node_stats_cmd(db_path: FilePath, input_dir: FilePath | None, output: FilePath) -> None:
    """Export one CSV row per locally recorded node-generation run.

    Token counts cover saved test_run and tool_generator model calls. Missing usage is blank.
    """
    from app.utils.export.node_generation_stats import export_node_stats
    from llm.utils.trajectory_store import get_trajectory_root

    trajectory_root = input_dir or get_trajectory_root()
    try:
        count = export_node_stats(db_path, trajectory_root, output, show_progress=True)
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    logger.success('Exported {} node-generation runs to {}', count, output.expanduser().resolve())


@cli.command(name='export-test-run-reports')
@click.option(
    '--db-path',
    type=click.Path(dir_okay=False, path_type=FilePath),
    default=DB_PATH,
    show_default=True,
    help='DuckDB file containing tool_functions run-to-tool references.',
)
@click.option(
    '--input-dir',
    type=click.Path(file_okay=False, path_type=FilePath),
    default=None,
    help='Saved agent trajectories (default: persistent.storage.storage_path/agent_trajectories).',
)
@click.option(
    '--output',
    type=click.Path(dir_okay=False, path_type=FilePath),
    default=FilePath('presets/build/test_run_reports.csv'),
    show_default=True,
    help='Destination CSV mapping tool_<index>_<name> lookup IDs to report text.',
)
def export_test_run_reports_cmd(db_path: FilePath, input_dir: FilePath | None, output: FilePath) -> None:
    """Export searchable report text for recorded tool entries."""
    from app.utils.export.test_run_reports import export_test_run_reports
    from llm.utils.trajectory_store import get_trajectory_root

    trajectory_root = input_dir or get_trajectory_root()
    try:
        result = export_test_run_reports(trajectory_root, db_path, output, show_progress=True)
    except (OSError, TypeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    logger.success(
        'Exported {} report lookups from {} saved reports ({} trajectory runs, {} events) to {}',
        result.lookup_count,
        result.source_report_count,
        result.run_count,
        result.event_count,
        result.output,
    )
    logger.info(
        'Resolved {} no-test-run references; skipped {} tool records without a recorded source report.',
        result.resolved_reference_count,
        result.missing_lookup_count,
    )


@cli.command(name='compact-trajectories')
@click.option(
    '--input-dir',
    type=click.Path(exists=True, file_okay=False, path_type=str),
    default=None,
    help='Immutable JSON trajectory root (default: persistent.storage.storage_path/agent_trajectories).',
)
@click.option(
    '--output-dir',
    type=click.Path(file_okay=False, path_type=str),
    default=None,
    help='Parquet output directory (default: persistent.storage.storage_path/agent_datasets/trajectories).',
)
@click.option('--agent', 'agents', multiple=True, help='Include only the specified agent; may be repeated.')
@click.option('--skip-invalid', is_flag=True, help='Skip invalid JSON files and record them in metadata.json.')
@click.option('--overwrite', is_flag=True, help='Atomically replace an existing output directory.')
def compact_trajectories_cmd(
    input_dir: str | None,
    output_dir: str | None,
    agents: tuple[str, ...],
    *,
    skip_invalid: bool,
    overwrite: bool,
) -> None:
    """Compact raw agent JSON events into events/runs Parquet datasets."""
    from llm.utils.trajectory_store import get_trajectory_root

    resolved_input = input_dir or str(get_trajectory_root())
    resolved_output = output_dir or _default_trajectory_dataset_path()
    compact_trajectories(resolved_input, resolved_output, agents, skip_invalid=skip_invalid, overwrite=overwrite)


if __name__ == '__main__':
    cli()
