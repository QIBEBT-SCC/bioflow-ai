import asyncio
import os
import shutil
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

if TYPE_CHECKING:
    from langgraph.checkpoint.redis import AsyncRedisSaver
    from langgraph.graph.state import CompiledStateGraph

    from app.schemas.tool_config import ImageConfig


DB_PATH = 'presets/build/develop.db'
TRAJECTORY_DATASET_PATH = 'storage/agent_datasets/trajectories'
TRAJECTORY_INSERT_BATCH_SIZE = 500


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
        msg = f'轨迹目录不存在: {source_root}'
        raise click.ClickException(msg)
    if destination.exists() and not overwrite:
        msg = f'输出目录已存在: {destination}。如需替换, 请添加 --overwrite。'
        raise click.ClickException(msg)

    event_files = sorted(source_root.glob('agent=*/run=*/events/*.json'))
    if not event_files:
        msg = f'没有找到轨迹事件: {source_root}'
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
                    msg = f'无效轨迹文件 {event_file}: {error}'
                    raise click.ClickException(msg) from error
        if batch:
            con.executemany(insert_sql, batch)

        if selected_count == 0:
            selected_agents = ', '.join(agents) if agents else '(all)'
            msg = f'没有符合 agent 过滤条件的事件: {selected_agents}'
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
        f'轨迹整合完成: {selected_count} 个事件, {run_count} 个运行, '
        f'{len(invalid_files)} 个无效文件; 输出目录: {destination}'
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


async def _cleanup_threads(con: Any, thread_ids: list[str], indices: list[int]) -> None:
    """清理指定线程的检查点、测试记录及文件, 并重置DuckDB状态."""
    from app.core.config import get_settings
    from app.utils.container import remove_container
    from llm.utils.session import delete_checkpoint

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
                        logger.info('清理了残余容器')
                    except Exception as e:
                        logger.warning(e)
                await delete_checkpoint(checkpoint, thread_id)
                logger.info(f'删除了智能体记录: {thread_id}')

        placeholders = ','.join(['?' for _ in thread_ids])
        deleted = con.execute(
            f'DELETE FROM test_run WHERE test_run_id IN ({placeholders})',  # noqa: S608
            thread_ids,
        ).rowcount
        if deleted:
            logger.info(f'从 test_run 表中删除了 {deleted} 条保存数据')

        for thread_id in thread_ids:
            output_path = await Path(get_settings().persistent.storage.STORAGE_PATH).absolute() / 'testrun' / thread_id
            if await output_path.exists():
                shutil.rmtree(output_path)
                logger.info(f'清理了运行目录: {output_path}')

    if indices:
        idx_placeholders = ','.join(['?' for _ in indices])
        con.execute(
            f'UPDATE tool_functions SET thread = NULL, tool_node = NULL, interrupted = false WHERE index IN ({idx_placeholders})',  # noqa: S608
            indices,
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
        content.append('Interrupted: ', style='bold cyan')
        if row.get('interrupted', False):
            content.append(f'{row["interrupted"]}\n', style='red')
        else:
            content.append(f'{row["interrupted"]}\n', style='green')
            content.append('Node ID: ', style='bold cyan')
            content.append(f'{row["tool_node"]}\n', style='white')
        content.append('Description:\n', style='bold cyan')
        content.append(row['function_description'], style='white')

        console.print(
            Panel(
                content,
                title=f'[yellow]#{row["index"]} {row["thread"]}[/yellow]',
                border_style='blue',
                expand=True,
            )
        )

    con.close()


async def init_tools(index: int | None = None) -> None:
    """初始化工具节点."""
    import sys

    import duckdb
    import rich
    from langchain_core.messages import HumanMessage
    from rich.console import Console

    from llm.context import CreateToolContext
    from llm.utils.render import print_stream_chunk

    if TYPE_CHECKING:
        from langchain_core.runnables import RunnableConfig

    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_PROJECT'] = 'build-tool-node'

    skip_tools = ['MACS2', 'RSEM', 'lima', 'awk', 'PhyML', 'Snakemake', 'MrBayes']

    con = duckdb.connect(DB_PATH)

    if index is not None:
        todo_rows = _rows_to_dicts(con, 'SELECT * FROM tool_functions WHERE index = ?', [index])
        if not todo_rows:
            logger.error(f'未找到索引为 {index} 的数据')
            con.close()
            return
    else:
        placeholders = ','.join(['?' for _ in skip_tools])
        todo_rows = _rows_to_dicts(
            con,
            f'SELECT * FROM tool_functions WHERE tool_name NOT IN ({placeholders}) AND thread IS NULL',  # noqa: S608
            skip_tools,
        )

    console = Console()

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

            thread_id = f'tool_{row_index}_{tool_name}'
            config: RunnableConfig = {'configurable': {'thread_id': thread_id}, 'recursion_limit': sys.maxsize}
            context = CreateToolContext(
                user_id=2,
                skip_search=False,
                auto_save=True,
                save_message=True,
                test_run_id=thread_id,
            )

            con.execute('UPDATE tool_functions SET thread = ? WHERE index = ?', [thread_id, row_index])

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
                con.execute('UPDATE tool_functions SET interrupted = true WHERE index = ?', [row_index])
                logger.error(f'{thread_id} interrupted')
                rich.print(response.interrupts)
            else:
                con.execute(
                    'UPDATE tool_functions SET tool_node = ? WHERE index = ?',
                    [response.values['tool_uid'], row_index],
                )
                logger.success(f'{thread_id} done')

    con.close()


async def create_tool_manually(name: str, help_command: str, function: str, thread_id: str, *, drop_old: bool) -> None:
    """手动模拟工具生成."""
    import sys

    import rich
    from langchain_core.messages import HumanMessage
    from rich.console import Console

    from llm.context import CreateToolContext
    from llm.utils.render import print_stream_chunk

    if TYPE_CHECKING:
        from langchain_core.runnables import RunnableConfig

    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_PROJECT'] = 'build-tool-node'

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
        logger.error(f'未找到索引为 {index} 的数据')
        return

    async with _tool_agent_session() as (_, agent):
        thread_id = tool_info['thread']
        config: RunnableConfig = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 500}

        response = await agent.aget_state(config)
        rich.print(response.interrupts)


async def continue_run(index: int, message: str | None = None, *, approve: bool = False) -> None:
    """响应中断信息."""
    import duckdb
    import rich
    from langgraph.types import Command
    from rich.console import Console

    from llm.context import CreateToolContext
    from llm.utils.render import print_stream_chunk

    if TYPE_CHECKING:
        from langchain_core.runnables import RunnableConfig

    con = duckdb.connect(DB_PATH)
    con.execute('UPDATE tool_functions SET interrupted = false WHERE index = ?', [index])
    tool_info = _row_to_dict(con, 'SELECT * FROM tool_functions WHERE index = ?', [index])

    if tool_info is None:
        logger.error(f'未找到索引为 {index} 的数据')
        con.close()
        return

    console = Console()

    async with _tool_agent_session() as (_, agent):
        thread_id = tool_info['thread']
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
        if response.interrupts:
            con.execute('UPDATE tool_functions SET interrupted = true WHERE index = ?', [index])
            rich.print(response.interrupts)
        else:
            con.execute(
                'UPDATE tool_functions SET tool_node = ? WHERE index = ?',
                [response.values['tool_uid'], index],
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
                logger.warning(f'未找到索引为 {index} 的数据, 跳过')
                continue
            logger.info(f'要删除的数据 (index={index}): {tool_info}')
            valid_indices.append(index)

            if tool_info['tool_node']:
                with Session(engine) as session:
                    tool_crud.delete(session, tool_info['tool_node'])
                    session.commit()
                    logger.info(f'删除了一条数据库信息 (index={index})')
                await vectorstore.adelete(
                    filter_expression='tool_uid == {uid}',
                    filter_params={'uid': tool_info['tool_node']},
                )

            if tool_info['thread']:
                thread_ids_to_delete.append(tool_info['thread'])

    await _cleanup_threads(con, thread_ids_to_delete, valid_indices)
    con.close()
    logger.info(f'成功删除 {len(valid_indices)} 个工具记录')


async def check_records(*, clean_interrupted: bool = False, dry_run: bool = False) -> None:
    """检查并重置孤立的DuckDB记录."""
    import duckdb

    import app.crud.tools.tool as tool_crud
    from app.core.session import engine

    con = duckdb.connect(DB_PATH)
    indices_to_reset: list[int] = []
    thread_ids_to_delete: list[str] = []

    rows_with_node = _rows_to_dicts(con, 'SELECT * FROM tool_functions WHERE tool_node IS NOT NULL')
    with Session(engine) as session:
        for row in rows_with_node:
            if tool_crud.get(session, row['tool_node']) is None:
                indices_to_reset.append(row['index'])
                if row['thread']:
                    thread_ids_to_delete.append(row['thread'])
                logger.info(f'发现孤立记录 (index={row["index"]}): tool_node={row["tool_node"]} 在数据库中不存在')

    if clean_interrupted:
        interrupted_rows = _rows_to_dicts(con, 'SELECT * FROM tool_functions WHERE interrupted = true')
        for row in interrupted_rows:
            if row['index'] not in indices_to_reset:
                indices_to_reset.append(row['index'])
                if row['thread']:
                    thread_ids_to_delete.append(row['thread'])
                logger.info(f'发现中断记录 (index={row["index"]}): {row["tool_name"]}')

    if not indices_to_reset:
        logger.info('没有需要重置的记录')
        con.close()
        return

    if dry_run:
        logger.info(
            f'[DRY RUN] 共发现 {len(indices_to_reset)} 条待重置记录, 涉及 {len(thread_ids_to_delete)} 条智能体记录'
        )
        logger.info(f'[DRY RUN] 待重置索引: {indices_to_reset}')
        con.close()
        return

    await _cleanup_threads(con, thread_ids_to_delete, indices_to_reset)
    con.close()
    logger.info(f'成功重置 {len(indices_to_reset)} 条记录')


def export_images(*, upload: bool = False) -> None:
    """导出镜像数据."""
    import polars as pl

    from app.core.config import get_settings

    db_sdn = get_settings().persistent.database.DB_URL

    image_df = pl.read_database_uri(
        'SELECT * FROM images',
        uri=db_sdn.unicode_string().replace(db_sdn.scheme, db_sdn.scheme.split('+')[0]),
        engine='connectorx',
    ).sort('uid')
    logger.info(f'共计导出了 {image_df.height} 条数据')
    image_df.write_parquet('presets/images.parquet', compression='snappy')

    if upload:
        from huggingface_hub import batch_bucket_files

        batch_bucket_files(
            'Aye10032/bioflow-ai-prebuild',
            add=[('presets/images.parquet', 'images.parquet')],
        )


def export_docs(*, upload: bool = False) -> None:
    """导出工具文档."""
    import polars as pl

    from app.core.config import get_settings

    db_sdn = get_settings().persistent.database.DB_URL

    doc_df = pl.read_database_uri(
        'SELECT * FROM tool_docs',
        uri=db_sdn.unicode_string().replace(db_sdn.scheme, db_sdn.scheme.split('+')[0]),
        engine='connectorx',
    ).sort('uid')
    logger.info(f'共计导出了 {doc_df.height} 条数据')

    doc_df.write_parquet('presets/tool_docs.parquet', compression='snappy')

    if upload:
        from huggingface_hub import batch_bucket_files

        batch_bucket_files(
            'Aye10032/bioflow-ai-prebuild',
            add=[('presets/tool_docs.parquet', 'tool_docs.parquet')],
        )


def export_tools(*, upload: bool = False) -> None:
    """导出工具节点."""
    import polars as pl

    from app.core.config import get_settings

    db_sdn = get_settings().persistent.database.DB_URL

    tool_df = pl.read_database_uri(
        """
            SELECT t.*,
                   COALESCE((
                       SELECT jsonb_agg(jsonb_build_object('id', tt.id, 'name', tt.name) ORDER BY tt.id)
                       FROM tool_tag_link ttl
                       JOIN tool_tags tt ON tt.id = ttl.tag_id
                       WHERE ttl.tool_uid = t.uid
                   ), '[]'::jsonb)::text AS tags
            FROM tools t
            """,
        uri=db_sdn.unicode_string().replace(db_sdn.scheme, db_sdn.scheme.split('+')[0]),
        engine='connectorx',
    ).sort('uid')
    logger.info(f'共计导出了 {tool_df.height} 条数据')

    tool_df.write_parquet('presets/tools.parquet', compression='snappy')

    if upload:
        from huggingface_hub import batch_bucket_files

        batch_bucket_files(
            'Aye10032/bioflow-ai-prebuild',
            add=[('presets/tools.parquet', 'tools.parquet')],
        )


def statistic() -> None:
    """统计生成情况."""
    import duckdb
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    con = duckdb.connect(DB_PATH, read_only=True)

    total = (con.execute('SELECT COUNT(*) FROM tool_functions').fetchone() or (0,))[0]
    processed = (con.execute('SELECT COUNT(*) FROM tool_functions WHERE thread IS NOT NULL').fetchone() or (0,))[0]
    success = (con.execute('SELECT COUNT(*) FROM tool_functions WHERE tool_node IS NOT NULL').fetchone() or (0,))[0]
    unique_nodes = (
        con.execute('SELECT COUNT(DISTINCT tool_node) FROM tool_functions WHERE tool_node IS NOT NULL').fetchone()
        or (0,)
    )[0]
    failed = (
        con.execute('SELECT COUNT(*) FROM tool_functions WHERE thread IS NOT NULL AND tool_node IS NULL').fetchone()
        or (0,)
    )[0]
    interrupted = (con.execute('SELECT COUNT(*) FROM tool_functions WHERE interrupted = true').fetchone() or (0,))[0]

    console = Console()

    stats = Table.grid(padding=(0, 2))
    stats.add_column(style='bold', justify='right')
    stats.add_column()

    pct = (processed / total * 100) if total else 0

    stats.add_row('总记录数', f'[cyan]{total:,}[/cyan]')
    stats.add_row('已处理', f'[cyan]{processed:,}[/cyan] [dim]({pct:.1f}%)[/dim]')
    stats.add_row('成功', f'[green]{success:,}[/green]  生成节点 [green]{unique_nodes:,}[/green] 个')
    stats.add_row('失败', f'[red]{failed:,}[/red]  其中中断 [yellow]{interrupted:,}[/yellow] 条')

    console.print(
        Panel(
            stats,
            title='[bold]运行统计[/bold]',
            border_style='blue',
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )

    def _render_record_panel(row: dict | None, title: str, border_style: str = 'cyan') -> None:
        if not row:
            console.print(Panel('[dim](暂无数据)[/dim]', title=title, border_style=border_style))
            return

        body = Table.grid(padding=(0, 2))
        body.add_column(style='bold cyan', justify='right', no_wrap=True)
        body.add_column(overflow='fold')

        for key, value in row.items():
            if value is None:
                display = '[dim]None[/dim]'
            elif isinstance(value, bool):
                display = f'[red]{value}[/red]' if value else f'[green]{value}[/green]'
                if key == 'interrupted' and value:
                    display = f'[bold red]{value}[/bold red]'
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

    latest = _row_to_dict(con, 'SELECT * FROM tool_functions WHERE thread IS NOT NULL ORDER BY index DESC LIMIT 1')
    _render_record_panel(latest, '[bold green]最新运行记录[/bold green]')

    latest_interrupt = _row_to_dict(con, 'SELECT * FROM tool_functions WHERE interrupted = true ORDER BY index LIMIT 1')
    _render_record_panel(latest_interrupt, '[bold yellow]最新中断记录[/bold yellow]')

    con.close()


@click.group()
def cli() -> None:
    """开发人员使用工具."""
    from app.utils.logger import init_logging

    init_logging()


@cli.command()
@click.option('--name', '-n', type=str, default=None, help='要搜索的工具名称')
def search_cmd(name: str) -> None:
    """继续运行指定索引的工具创建任务."""
    search_tools(tool_name=name)


@cli.command()
@click.option('--index', type=int, default=None, help='指定要处理的数据索引 (如果不指定则处理所有未完成的数据)')
@click.option('--name', '-n', help='工具名称; 提供时切换到手动创建模式')
@click.option('--help-command', '-c', help='获取工具帮助文档的命令 (手动模式必填)')
@click.option('--function', '-f', help='需要生成的工具功能描述 (手动模式必填)')
@click.option('--thread-id', '-t', help='检查点线程 ID (手动模式必填)')
@click.option('--drop-old', is_flag=True, default=False, help='运行前删除该线程的旧检查点')
def create_tools_cmd(
    index: int | None,
    name: str | None,
    help_command: str | None,
    function: str | None,
    thread_id: str | None,
    *,
    drop_old: bool,
) -> None:
    """自动创建工具, 或使用 --name 手动创建单个工具."""
    manual_values = {
        '--name': name,
        '--help-command': help_command,
        '--function': function,
        '--thread-id': thread_id,
    }
    if any(value is not None for value in manual_values.values()):
        missing = [option for option, value in manual_values.items() if value is None]
        if missing:
            message = f'手动创建模式缺少参数: {", ".join(missing)}'
            raise click.UsageError(message)
        if index is not None:
            message = '--index 不能与手动创建参数同时使用'
            raise click.UsageError(message)
        # The completeness check above narrows these values at runtime.
        asyncio.run(
            create_tool_manually(
                cast('str', name),
                cast('str', help_command),
                cast('str', function),
                cast('str', thread_id),
                drop_old=drop_old,
            )
        )
        return

    if drop_old:
        message = '--drop-old 仅能用于手动创建模式'
        raise click.UsageError(message)
    asyncio.run(init_tools(index))


@cli.command()
@click.argument('index', type=int)
def read_interrupt_cmd(index: int) -> None:
    """读取工具中断信息."""
    asyncio.run(read_interrupt(index))


@cli.command()
@click.argument('index', type=int)
@click.option('--message', '-m', type=str, default=None, help='继续运行时发送的消息')
@click.option('--approve', is_flag=True, default=False, help='批准标志')
def continue_run_cmd(index: int, message: str | None, *, approve: bool) -> None:
    """继续运行指定索引的工具创建任务."""
    asyncio.run(continue_run(index, message, approve=approve))


@cli.command()
@click.argument('indices', type=int, nargs=-1, required=True)
def delete_prebuild_tool_cmd(indices: tuple[int, ...]) -> None:
    """删除指定索引的预构建工具 (可一次删除多个, 例如: 61 62 63 65)."""
    asyncio.run(delete_prebuild_tool(list(indices)))


@cli.command()
@click.option('--clean-interrupted', is_flag=True, default=False, help='同时清理 interrupted=true 的记录')
@click.option('--dry-run', is_flag=True, default=False, help='仅预览待处理记录, 不执行实际操作')
def check_cmd(*, clean_interrupted: bool, dry_run: bool) -> None:
    """检查并重置孤立的 DuckDB 记录."""
    asyncio.run(check_records(clean_interrupted=clean_interrupted, dry_run=dry_run))


@cli.command()
@click.option('--upload', is_flag=True, default=False, help='是否同步到huggingface')
@click.option(
    '--db',
    type=click.Choice(['images', 'docs', 'tools']),
    multiple=True,
    default=['images', 'docs', 'tools'],
    show_default=True,
    help='选择要导出的数据库, 可多选',
)
def export(*, upload: bool, db: tuple[str, ...]) -> None:
    """导出数据库数据."""
    db_map = {
        'images': export_images,
        'docs': export_docs,
        'tools': export_tools,
    }
    for name in db:
        logger.info(f'正在导出 {name}...')
        db_map[name](upload=upload)


@cli.command()
def statistic_cmd() -> None:
    """显示节点生成统计数据."""
    statistic()


@cli.command(name='compact-trajectories')
@click.option(
    '--input-dir',
    type=click.Path(exists=True, file_okay=False, path_type=str),
    default='storage/agent_trajectories',
    show_default=True,
    help='不可变 JSON 轨迹的根目录',
)
@click.option(
    '--output-dir',
    type=click.Path(file_okay=False, path_type=str),
    default=TRAJECTORY_DATASET_PATH,
    show_default=True,
    help='Parquet 数据集输出目录',
)
@click.option('--agent', 'agents', multiple=True, help='只整合指定 agent; 可重复传入')
@click.option('--skip-invalid', is_flag=True, help='跳过损坏或字段不完整的 JSON, 并记录到 metadata.json')
@click.option('--overwrite', is_flag=True, help='原子替换已存在的输出目录')
def compact_trajectories_cmd(
    input_dir: str,
    output_dir: str,
    agents: tuple[str, ...],
    *,
    skip_invalid: bool,
    overwrite: bool,
) -> None:
    """将原始智能体 JSON 事件整合为 events/runs Parquet 数据集."""
    compact_trajectories(input_dir, output_dir, agents, skip_invalid=skip_invalid, overwrite=overwrite)


if __name__ == '__main__':
    cli()
