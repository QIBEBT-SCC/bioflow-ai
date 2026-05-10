import asyncio
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from loguru import logger

DB_PATH = 'presets/build/develop.db'


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
    import duckdb
    import rich
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.redis import AsyncRedisSaver
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
    from phoenix.otel import register
    from rich.console import Console
    from sqlmodel import Session

    import app.crud.tools.image as image_crud
    import app.crud.tools.tool_doc as doc_crud
    from app.core.config import get_settings
    from app.core.session import engine
    from app.models.tools import ToolHelpDoc
    from app.utils.container import run_command
    from llm.agents.tool_agent import CreateToolAgent
    from llm.states import CreateToolContext
    from llm.utils.render import print_stream_chunk

    if TYPE_CHECKING:
        from langchain_core.runnables import RunnableConfig

    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_PROJECT'] = 'build-tool-node'

    # configure the Phoenix tracer
    tracer_provider = register(  # noqa: F841
        endpoint=get_settings().develop.PHOENIX_ENDPOINT,
        project_name='build-tool',
        batch=True,
        auto_instrument=True,
        set_global_tracer_provider=False,
    )

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

    async with AsyncRedisSaver.from_conn_string(get_settings().develop.CHECKPOINT_URL.unicode_string()) as checkpoint:
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
        await checkpoint.setup()

        agent = CreateToolAgent().build(memory=checkpoint)

        for row in todo_rows:
            row_index: int = row['index']
            tool_name: str = row['tool_name']
            function: str = row['function_description']
            help_command: str = row['help_command'].strip()

            with Session(engine) as session:
                tool_image = image_crud.get_by_name(session, tool_name)
                assert tool_image is not None  # noqa: S101

                tool_doc = doc_crud.get_by_command(session, help_command)
                if tool_doc is None:
                    code, result = run_command(tool_image.image, [], help_command)
                    tool_doc = ToolHelpDoc(help_command=help_command, content=result, image_uid=tool_image.uid)
                    tool_doc = doc_crud.insert(session, tool_doc)

                agent_input = {
                    'messages': [
                        HumanMessage(
                            content=f'<Require>\n'
                            f'tool_name: {tool_name}\n'
                            f'function: {function}\nhelp_command: `{help_command}`\n'
                            f'</Require>'
                        )
                    ],
                    'image_uid': tool_image.uid,
                    'image': tool_image.image,
                    'doc_uid': tool_doc.uid,
                }

            thread_id = f'tool_{row_index}_{tool_name}'
            config: RunnableConfig = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 2000}
            context = CreateToolContext(user_id=2, auto_save=True, save_message=True, test_run_id=thread_id)

            con.execute('UPDATE tool_functions SET thread = ? WHERE index = ?', [thread_id, row_index])

            async for chunk in agent.astream(
                agent_input,  # type: ignore
                config,
                context=context,  # type: ignore
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
    from langgraph.checkpoint.redis import AsyncRedisSaver
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
    from phoenix.otel import register
    from rich.console import Console
    from sqlmodel import Session

    import app.crud.tools.image as image_crud
    import app.crud.tools.tool_doc as doc_crud
    from app.core.config import get_settings
    from app.core.session import engine
    from app.models.tools import ToolHelpDoc
    from app.utils.container import run_command
    from llm.agents.tool_agent.tool_creator import CreateToolAgent
    from llm.states import CreateToolContext, CreateToolState
    from llm.utils.render import print_stream_chunk

    if TYPE_CHECKING:
        from langchain_core.runnables import RunnableConfig

    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_PROJECT'] = 'build-tool-node'

    # configure the Phoenix tracer
    tracer_provider = register(  # noqa: F841
        endpoint=get_settings().develop.PHOENIX_ENDPOINT,
        project_name='build-tool',
        batch=True,
        auto_instrument=True,
        set_global_tracer_provider=False,
    )

    async with AsyncRedisSaver.from_conn_string(get_settings().develop.CHECKPOINT_URL.unicode_string()) as checkpoint:
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
            await checkpoint.adelete_thread(thread_id)

        agent = CreateToolAgent().build(memory=checkpoint)

        with Session(engine) as session:
            tool_image = image_crud.get_by_name(session, name)
            if tool_image is None:
                logger.error('Image not found.')
                return

            tool_doc = doc_crud.get_by_command(session, help_command)
            if tool_doc is None:
                code, result = run_command(tool_image.image, [], help_command)
                tool_doc = ToolHelpDoc(help_command=help_command, content=result, image_uid=tool_image.uid)
                tool_doc = doc_crud.insert(session, tool_doc)

            agent_input: CreateToolState = {
                'messages': [
                    HumanMessage(
                        content=f'<Require>\n'
                        f'tool_name: {name}\n'
                        f'function: {function}\n'
                        f'help_command: `{help_command}`\n'
                        f'</Require>'
                    )
                ],
                'image_uid': tool_image.uid,
                'image': tool_image.image,
                'doc_uid': tool_doc.uid,
            }

        config: RunnableConfig = {'configurable': {'thread_id': thread_id}, 'recursion_limit': sys.maxsize}
        context = CreateToolContext(user_id=2, auto_save=True, test_run_id=thread_id)

        console = Console()
        async for chunk in agent.astream(
            agent_input,  # type: ignore
            config,
            context=context,  # type: ignore
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
    from langgraph.checkpoint.redis import AsyncRedisSaver
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

    from app.core.config import get_settings
    from llm.agents.tool_agent import CreateToolAgent

    if TYPE_CHECKING:
        from langchain_core.runnables import RunnableConfig

    con = duckdb.connect(DB_PATH)
    tool_info = _row_to_dict(con, 'SELECT * FROM tool_functions WHERE index = ?', [index])
    con.close()

    if tool_info is None:
        logger.error(f'未找到索引为 {index} 的数据')
        return

    async with AsyncRedisSaver.from_conn_string(get_settings().develop.CHECKPOINT_URL.unicode_string()) as checkpoint:
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

        agent = CreateToolAgent().build(memory=checkpoint)

        thread_id = tool_info['thread']
        config: RunnableConfig = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 500}

        response = await agent.aget_state(config)
        rich.print(response.interrupts)


async def continue_run(index: int, message: str | None = None, *, approve: bool = False) -> None:
    """响应中断信息."""
    import duckdb
    import rich
    from langgraph.checkpoint.redis import AsyncRedisSaver
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
    from langgraph.types import Command
    from rich.console import Console

    from app.core.config import get_settings
    from llm.agents.tool_agent import CreateToolAgent
    from llm.states import CreateToolContext
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

    async with AsyncRedisSaver.from_conn_string(get_settings().develop.CHECKPOINT_URL.unicode_string()) as checkpoint:
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

        agent = CreateToolAgent().build(memory=checkpoint)

        thread_id = tool_info['thread']
        config: RunnableConfig = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 500}
        context = CreateToolContext(user_id=2, auto_save=True, save_message=True, test_run_id=thread_id)

        if approve:
            input_message = Command(resume=True)
        elif message is not None:
            input_message = Command(resume=message)
        else:
            input_message = None
        async for chunk in agent.astream(
            input_message,
            config,
            context=context,  # type: ignore
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
    from langgraph.checkpoint.redis import AsyncRedisSaver
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
    from sqlmodel import Session

    import app.crud.tools.tool as tool_crud
    from app.core.config import get_settings
    from app.core.session import engine
    from app.utils.container import remove_container
    from llm.agents.tool_agent import CreateToolAgent

    if TYPE_CHECKING:
        from langchain_core.runnables import RunnableConfig

    con = duckdb.connect(DB_PATH)
    thread_ids_to_delete: list[str] = []

    for index in indices:
        tool_info = _row_to_dict(con, 'SELECT * FROM tool_functions WHERE index = ?', [index])
        if tool_info is None:
            logger.warning(f'未找到索引为 {index} 的数据, 跳过')
            continue
        logger.info(f'要删除的数据 (index={index}): {tool_info}')

        if tool_info['tool_node']:
            with Session(engine) as session:
                tool_crud.delete(session, tool_info['tool_node'])
                logger.info(f'删除了一条数据库信息 (index={index})')

        if tool_info['thread']:
            thread_ids_to_delete.append(tool_info['thread'])
            async with AsyncRedisSaver.from_conn_string(
                get_settings().develop.CHECKPOINT_URL.unicode_string()
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

                agent = CreateToolAgent().build(memory=checkpoint)

                config: RunnableConfig = {'configurable': {'thread_id': tool_info['thread']}, 'recursion_limit': 500}

                response = await agent.aget_state(config, subgraphs=True)
                try:
                    container_id = response.tasks[0].state.values['container_id']
                except (IndexError, AttributeError, KeyError, TypeError):
                    container_id = None
                if container_id:
                    try:
                        remove_container(response.tasks[0].state.values['container_id'])
                        logger.info('清理了残余容器')
                    except Exception as e:
                        logger.warning(e)

                await checkpoint.adelete_thread(tool_info['thread'])
                logger.info(f'删除了一条智能体记录 (index={index})')

    if thread_ids_to_delete:
        placeholders = ','.join(['?' for _ in thread_ids_to_delete])
        deleted = con.execute(
            f'DELETE FROM test_run WHERE test_run_id IN ({placeholders})',  # noqa: S608
            thread_ids_to_delete,
        ).rowcount
        if deleted:
            logger.info(f'从 test_run 表中删除了 {deleted} 条保存数据')

    for thread_id in thread_ids_to_delete:
        output_path = Path(get_settings().persistent.storage.STORAGE_PATH).absolute() / 'testrun' / thread_id
        if output_path.exists():
            shutil.rmtree(output_path)
            logger.info(f'清理了运行目录: {output_path}')

    if indices:
        idx_placeholders = ','.join(['?' for _ in indices])
        con.execute(
            f'UPDATE tool_functions SET thread = NULL, tool_node = NULL, interrupted = false WHERE index IN ({idx_placeholders})',  # noqa: S608
            indices,
        )
    con.close()
    logger.info(f'成功删除 {len(indices)} 个工具记录')


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

    tool_df = (
        pl.read_database_uri(
            """
        SELECT t.*,
               COALESCE(
                   json_agg(json_build_object('id', tt.id, 'name', tt.name) ORDER BY tt.id)
                   FILTER (WHERE tt.id IS NOT NULL),
                   '[]'
               )::text AS tags
        FROM tools t
        LEFT JOIN tool_tag_link ttl ON t.uid = ttl.tool_uid
        LEFT JOIN tool_tags tt ON ttl.tag_id = tt.id
        GROUP BY t.uid
            """,
            uri=db_sdn.unicode_string().replace(db_sdn.scheme, db_sdn.scheme.split('+')[0]),
            engine='connectorx',
        )
        # .with_columns(pl.col('tool_type').replace({'DOCKER': 0, 'CONDA': 1}).cast(pl.Int32))
        .sort('uid')
    )
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

    total = con.execute('SELECT COUNT(*) FROM tool_functions').fetchone()[0]  # type: ignore[index]
    processed = con.execute('SELECT COUNT(*) FROM tool_functions WHERE thread IS NOT NULL').fetchone()[0]  # type: ignore[index]
    success = con.execute('SELECT COUNT(*) FROM tool_functions WHERE tool_node IS NOT NULL').fetchone()[0]  # type: ignore[index]
    unique_nodes = con.execute(
        'SELECT COUNT(DISTINCT tool_node) FROM tool_functions WHERE tool_node IS NOT NULL'
    ).fetchone()[0]  # type: ignore[index]
    failed = con.execute(
        'SELECT COUNT(*) FROM tool_functions WHERE thread IS NOT NULL AND tool_node IS NULL'
    ).fetchone()[0]  # type: ignore[index]
    interrupted = con.execute('SELECT COUNT(*) FROM tool_functions WHERE interrupted = true').fetchone()[0]  # type: ignore[index]

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

    def _render_record_panel(row: dict, title: str, border_style: str = 'cyan') -> None:
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


@cli.command()
@click.option('--name', '-n', type=str, default=None, help='要搜索的工具名称')
def search_cmd(name: str) -> None:
    """继续运行指定索引的工具创建任务."""
    search_tools(tool_name=name)


@cli.command()
@click.option('--index', type=int, default=None, help='指定要处理的数据索引 (如果不指定则处理所有未完成的数据)')
def create_tools_cmd(index: int | None) -> None:
    """初始化工具."""
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


if __name__ == '__main__':
    cli()
