import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from loguru import logger
from tqdm import tqdm

# 禁用 crawl4ai 导致的 httpx 日志输出
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)


def _log_update(namespace: tuple[str], mode: str, data: dict[str, Any]) -> None:
    if mode == 'custom':
        logger.info(data)
    else:
        filtered_data = {k: v for k, v in data.items() if v is not None}
        if filtered_data:
            running_node = ','.join([n.split(':')[0] for n in namespace])
            logger.debug(f'({running_node}) {filtered_data}'[:120])


async def init_tools(index: int | None = None) -> None:
    """初始化工具节点."""
    import aiosqlite
    import polars as pl
    import rich
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from sqlmodel import Session

    import app.crud.tools.image as image_crud
    import app.crud.tools.tool_doc as doc_crud
    from app.core.session import engine
    from app.models.tools import ToolHelpDoc
    from app.utils.container import run_command
    from llm.agents.tool_manager import CreateToolAgent
    from llm.states import CreateToolContext

    if TYPE_CHECKING:
        from langchain_core.runnables import RunnableConfig

    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_PROJECT'] = 'build-tool-node'

    # 读取完整数据集
    function_df = pl.read_parquet('tool_functions_deepseek.parquet')

    # 需要跳过的工具 (不处理, 但保留在数据集中)
    skip_tools = ['MACS2', 'RSEM', 'lima', 'awk', 'PhyML', 'Snakemake', 'MrBayes']

    # 如果指定了 index, 只处理该索引的数据; 否则处理所有未完成的数据
    if index is not None:
        todo_df = function_df.filter(pl.col('index') == index)
        if todo_df.height == 0:
            logger.error(f'未找到索引为 {index} 的数据')
            return
    else:
        # 过滤掉跳过的工具和已完成的任务
        todo_df = function_df.filter(~pl.col('tool_name').is_in(skip_tools) & ~pl.col('thread').is_not_null())

    async with aiosqlite.connect('checkpoints.db') as conn:
        memory = AsyncSqliteSaver(conn, serde=JsonPlusSerializer(pickle_fallback=True))
        agent = CreateToolAgent().build(memory=memory)

        for row in todo_df.iter_rows(named=True):
            # Schema({'index': UInt32, 'tool_name': String, 'function_description': String, 'help_command': String, 'tool_node': Null, 'thread': Null, 'interrupted': Boolean})
            row: dict[str, Any]
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

            # 更新处理后的数据
            function_df = function_df.with_columns(
                pl.when(pl.col('index') == row_index)
                .then(pl.lit(thread_id))
                .otherwise(pl.col('thread'))
                .alias('thread')
            )
            function_df.write_parquet('tool_functions_deepseek.parquet', compression='snappy')

            async for namespace, mode, data in agent.astream(
                agent_input,  # type: ignore
                config,
                context=context,  # type: ignore
                stream_mode=['custom', 'updates'],
                subgraphs=True,
            ):
                _log_update(namespace, mode, data)

            response = await agent.aget_state(config)
            if response.interrupts:
                function_df = function_df.with_columns(
                    pl.when(pl.col('index') == row_index)
                    .then(pl.lit(True))  # noqa: FBT003
                    .otherwise(pl.col('interrupted'))
                    .alias('interrupted'),
                )
                function_df.write_parquet('tool_functions_deepseek.parquet', compression='snappy')

                logger.error(f'{thread_id} interrupted')
                rich.print(response.interrupts)
            else:
                function_df = function_df.with_columns(
                    pl.when(pl.col('index') == row_index)
                    .then(pl.lit(response.values['tool_uid']))
                    .otherwise(pl.col('tool_node'))
                    .alias('tool_node'),
                )
                function_df.write_parquet('tool_functions_deepseek.parquet', compression='snappy')
                logger.success(f'{thread_id} done')


async def read_interrupt(index: int) -> None:
    """加载指定中断信息."""
    import aiosqlite
    import polars as pl
    import rich
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from llm.agents.tool_manager import CreateToolAgent

    if TYPE_CHECKING:
        from langchain_core.runnables import RunnableConfig

    function_df = pl.read_parquet('tool_functions_deepseek.parquet')
    tool_info: dict[str, Any] = function_df.row(index=index, named=True)

    async with aiosqlite.connect('checkpoints.db') as conn:
        memory = AsyncSqliteSaver(conn, serde=JsonPlusSerializer(pickle_fallback=True))
        agent = CreateToolAgent().build(memory=memory)

        thread_id = tool_info['thread']
        config: RunnableConfig = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 500}

        response = await agent.aget_state(config)
        rich.print(response.interrupts)


async def continue_run(index: int, message: str | None = None, *, approve: bool = False) -> None:
    """响应中断信息."""
    import aiosqlite
    import polars as pl
    import rich
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.types import Command

    from llm.agents.tool_manager import CreateToolAgent
    from llm.states import CreateToolContext

    if TYPE_CHECKING:
        from langchain_core.runnables import RunnableConfig

    function_df = pl.read_parquet('tool_functions_deepseek.parquet')
    function_df = function_df.with_columns(
        pl.when(pl.col('index') == index)
        .then(pl.lit(value=False))
        .otherwise(pl.col('interrupted'))
        .alias('interrupted'),
    )
    tool_info: dict[str, Any] = function_df.filter(pl.col('index') == index).row(index=0, named=True)

    async with aiosqlite.connect('checkpoints.db') as conn:
        memory = AsyncSqliteSaver(conn, serde=JsonPlusSerializer(pickle_fallback=True))
        agent = CreateToolAgent().build(memory=memory)

        thread_id = tool_info['thread']
        config: RunnableConfig = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 500}
        context = CreateToolContext(user_id=2, auto_save=True, save_message=True, test_run_id=thread_id)

        if approve:
            input_message = Command(resume=True)
        elif message is not None:
            input_message = Command(resume=message)
        else:
            input_message = None
        async for namespace, mode, data in agent.astream(
            input_message,
            config,
            context=context,  # type: ignore
            stream_mode=['custom', 'updates'],
            subgraphs=True,
        ):
            _log_update(namespace, mode, data)

        response = await agent.aget_state(config)
        if response.interrupts:
            function_df = function_df.with_columns(
                pl.when(pl.col('index') == index)
                .then(pl.lit(value=True))
                .otherwise(pl.col('interrupted'))
                .alias('interrupted'),
            )
            rich.print(response.interrupts)
        else:
            function_df = function_df.with_columns(
                pl.when(pl.col('index') == index)
                .then(pl.lit(response.values['tool_uid']))
                .otherwise(pl.col('tool_node'))
                .alias('tool_node'),
            )
        function_df.write_parquet('tool_functions_deepseek.parquet', compression='snappy')


async def delete_prebuild_tool(indices: list[int]) -> None:
    """删除现有节点."""
    import aiosqlite
    import polars as pl
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from sqlmodel import Session

    import app.crud.tools.tool as tool_crud
    from app.core.config import get_settings
    from app.core.session import engine
    from app.utils.container import remove_container
    from llm.agents.tool_manager import CreateToolAgent

    if TYPE_CHECKING:
        from langchain_core.runnables import RunnableConfig

    function_df = pl.read_parquet('tool_functions_deepseek.parquet')
    thread_ids_to_delete: list[str] = []

    for index in indices:
        tool_info: dict[str, Any] = function_df.filter(pl.col('index') == index).row(index=0, named=True)
        logger.info(f'要删除的数据 (index={index}): {tool_info}')

        if tool_info['tool_node']:
            with Session(engine) as session:
                tool_crud.delete(session, tool_info['tool_node'])
                logger.info(f'删除了一条数据库信息 (index={index})')

        if tool_info['thread']:
            thread_ids_to_delete.append(tool_info['thread'])
            async with aiosqlite.connect('checkpoints.db') as conn:
                memory = AsyncSqliteSaver(conn, serde=JsonPlusSerializer(pickle_fallback=True))
                agent = CreateToolAgent().build(memory=memory)

                config: RunnableConfig = {'configurable': {'thread_id': tool_info['thread']}, 'recursion_limit': 500}

                response = await agent.aget_state(config, subgraphs=True)
                if (
                    response.tasks
                    and response.tasks[0].state
                    and 'container_id' in response.tasks[0].state.values
                    and response.tasks[0].state.values['container_id']
                ):
                    # 删除残余容器
                    try:
                        remove_container(response.tasks[0].state.values['container_id'])
                        logger.info('清理了残余容器')
                    except Exception as e:
                        logger.warning(e)

                await memory.adelete_thread(tool_info['thread'])
                logger.info(f'删除了一条智能体记录 (index={index})')

    # 批量删除 test_run.parquet 中的记录
    if thread_ids_to_delete and Path('test_run.parquet').exists():
        dataset_df = pl.read_parquet('test_run.parquet')
        original_count = dataset_df.height
        dataset_df = dataset_df.filter(~pl.col('test_run_id').is_in(thread_ids_to_delete))
        deleted_count = original_count - dataset_df.height
        if deleted_count > 0:
            dataset_df.write_parquet('test_run.parquet', compression='snappy')
            logger.info(f'从 test_run.parquet 中删除了 {deleted_count} 条保存数据')

    # 删除运行目录
    for thread_id in thread_ids_to_delete:
        output_path = Path(get_settings().persistent.storage.STORAGE_PATH).absolute() / 'testrun' / thread_id
        if output_path.exists():
            shutil.rmtree(output_path)
            logger.info(f'清理了运行目录: {output_path}')

    # 批量更新 DataFrame
    indices_condition = pl.col('index').is_in(indices)
    function_df = function_df.with_columns(
        pl.when(indices_condition).then(None).otherwise(pl.col('thread')).alias('thread'),
        pl.when(indices_condition).then(None).otherwise(pl.col('tool_node')).alias('tool_node'),
        pl.when(indices_condition).then(False).otherwise(pl.col('interrupted')).alias('interrupted'),  # noqa: FBT003
    )
    function_df.write_parquet('tool_functions_deepseek.parquet', compression='snappy')
    logger.info(f'成功删除 {len(indices)} 个工具记录')


def export_images(*, upload: bool = False) -> None:
    """导出镜像数据."""
    import polars as pl
    from sqlmodel import Session

    import app.crud.tools.image as image_crud
    import app.models.user  # noqa: F401
    from app.core.session import engine

    if TYPE_CHECKING:
        from app.models.tools import ToolImage

    batch_size = 100
    all_data = []

    with Session(engine) as session:
        total_count = image_crud.get_counts(session)

        for offset in tqdm(range(0, total_count, batch_size)):
            images: list[ToolImage] = image_crud.get_all(session, batch_size, offset)

            batch_data = [
                {
                    'uid': str(image.uid),
                    'name': image.name,
                    'version': image.version,
                    'description': image.description,
                    'homepage': image.homepage,
                    'paper_link': image.paper_link,
                    'image_url': image.image.get_image(),
                }
                for image in images
            ]
            all_data.extend(batch_data)

    output_df = pl.DataFrame(all_data)
    output_df.write_parquet('presets/images.parquet', compression='snappy')

    if upload:
        from huggingface_hub import batch_bucket_files

        batch_bucket_files(
            'Aye10032/bioflow-ai-prebuild',
            add=[('presets/images.parquet', 'images.parquet')],
        )


def export_docs(*, upload: bool = False) -> None:
    """导出工具文档."""
    import polars as pl
    from sqlmodel import Session

    import app.crud.tools.tool_doc as doc_crud
    import app.models.user  # noqa: F401
    from app.core.session import engine

    if TYPE_CHECKING:
        from app.models.tools import ToolHelpDoc

    batch_size = 100
    all_data = []

    with Session(engine) as session:
        total_count = doc_crud.get_counts(session)

        for offset in tqdm(range(0, total_count, batch_size)):
            docs: list[ToolHelpDoc] = doc_crud.get_all(session, batch_size, offset)

            batch_data = [
                {
                    'id': str(doc.uid),
                    'help_command': doc.help_command,
                    'description': doc.description if doc.description else '',
                    'content': doc.content,
                    'image_uid': str(doc.image_uid),
                }
                for doc in docs
            ]
            all_data.extend(batch_data)

    output_df = pl.DataFrame(all_data)
    output_df.write_parquet('presets/tool_docs.parquet', compression='snappy')

    if upload:
        from huggingface_hub import batch_bucket_files

        batch_bucket_files(
            'Aye10032/bioflow-ai-prebuild',
            add=[('presets/tool_docs.parquet', 'tool_docs.parquet')],
        )


def export_tools(*, upload: bool = False) -> None:
    """导出工具节点."""
    import polars as pl
    from sqlmodel import Session

    import app.crud.tools.tool as tool_crud
    import app.models.user  # noqa: F401
    from app.core.session import engine

    if TYPE_CHECKING:
        from app.models.tools import ToolDefine

    batch_size = 100
    all_data = []

    with Session(engine) as session:
        total_count = tool_crud.get_counts(session)

        for offset in tqdm(range(0, total_count, batch_size)):
            tools: list[ToolDefine] = tool_crud.get_all(session, batch_size, offset)

            batch_data = [
                {
                    'uid': str(tool.uid),
                    'name': tool.name,
                    'image_uid': str(tool.image_uid),
                    'description': tool.description,
                    'tool_type': tool.tool_type,
                    'group_id': tool.group_id,
                    'help_doc_uid': str(tool.help_doc_uid),
                    'command_template': tool.command_template,
                    'dynamic_params': [param.model_dump() for param in tool.dynamic_params],
                    'immutable_static_params': tool.immutable_static_params,
                    'modifiable_static_params': tool.modifiable_static_params,
                    'file_mounts': [output.model_dump() for output in tool.file_mounts],
                    'tags': [tag.model_dump() for tag in tool.tags],
                }
                for tool in tools
            ]
            all_data.extend(batch_data)

        if all_data:
            output_df = pl.DataFrame(all_data)
            output_df.write_parquet('presets/tools.parquet', compression='snappy')

    if upload:
        from huggingface_hub import batch_bucket_files

        batch_bucket_files(
            'Aye10032/bioflow-ai-prebuild',
            add=[('presets/tools.parquet', 'tools.parquet')],
        )


def statistic() -> None:
    """统计生成情况."""
    import polars as pl
    import rich

    function_df = pl.read_parquet('tool_functions_deepseek.parquet')
    print(f'共计 {function_df.height} 条记录')
    print(f'当前已处理 {function_df.filter(pl.col("thread").is_not_null()).height} 条数据')
    print(
        f'成功 {function_df.filter(pl.col("tool_node").is_not_null()).height} 条, 共计生成 {function_df["tool_node"].unique().count()} 个节点'
    )
    print(
        f'失败 {function_df.filter(pl.col("thread").is_not_null() & pl.col("tool_node").is_null()).height} 条, 其中中断 {function_df["interrupted"].sum()} 条'
    )
    print('=' * 50)
    print('最新运行记录为: ')
    rich.print(function_df.filter(pl.col('thread').is_not_null()).top_k(1, by='index').row(0, named=True))
    print('=' * 50)
    print('最新中断为: ')
    rich.print(function_df.filter(pl.col('interrupted')).sort('index', descending=False).head(1).row(0, named=True))


@click.group()
def cli() -> None:
    """开发人员使用工具."""


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
def show_statistic() -> None:
    """显示节点生成统计数据."""
    statistic()


if __name__ == '__main__':
    cli()
