import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import click
from loguru import logger

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

# 禁用 crawl4ai 导致的 httpx 日志输出
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)


def _clear_container() -> None:
    import docker

    client = docker.from_env()
    for container in client.containers.list():
        image_name = ''.join(container.image.tags)
        if 'biocontainers' in image_name:
            container.remove(force=True)


async def init_images(*, drop_old: bool = True) -> None:
    """刷新镜像向量索引."""
    import polars as pl
    from langchain_core.documents import Document
    from langchain_milvus import Milvus
    from sqlmodel import Session
    from tqdm import tqdm

    import app.crud.tools.image as image_crud
    from app.core.config import get_settings
    from app.core.session import engine
    from app.models.tools import ImageConfig, ToolImage
    from llm.models import qwen3_embedding

    with Session(engine) as session:
        if drop_old:
            delete_count = image_crud.drop_all(session)
            logger.info(f'Delete {delete_count} data')

        image_df = pl.read_parquet('../presets/images.parquet')

        images: list[ToolImage] = []
        for row in image_df.iter_rows(named=True):
            try:
                image = ImageConfig.from_str(row['image_url'])
                image_data = row
                image_data['image'] = image
                images.append(ToolImage.model_validate(image_data))
            except Exception as e:
                logger.warning(f'Invalid image URL: {row["image_url"]}, {e}')

        images = image_crud.insert_all(session, images)

    embedding = qwen3_embedding()
    vectorstore = Milvus(
        embedding_function=embedding,
        collection_name='ImageCollection',
        connection_args={'uri': get_settings().knowledge_base.MILVUS_URL},
        index_params={'index_type': 'HNSW', 'metric_type': 'L2'},
        search_params={'M': 64, 'efConstruction': 100},
        consistency_level='Strong',
        drop_old=drop_old,
        auto_id=True,
    )

    batch_size = 50

    for i in tqdm(range(0, len(images), batch_size)):
        batch_images = images[i : i + batch_size]

        docs = [
            Document(page_content=img.description, metadata={'image_uid': str(img.uid), 'name': img.name})
            for img in batch_images
        ]

        await vectorstore.aadd_documents(docs)


async def init_tools(index: Optional[int] = None) -> None:
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
                    .then(pl.lit(True))
                    .otherwise(pl.col('interrupted'))
                    .alias('interrupted'),
                )
                function_df.write_parquet('tool_functions_deepseek.parquet', compression='snappy')

                logger.error(f'{thread_id} interrupted')
                rich.print(response.interrupts)
                _clear_container()
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

    function_df = pl.read_parquet('tool_functions_deepseek.parquet')
    tool_info: dict[str, Any] = function_df.row(index=index, named=True)

    async with aiosqlite.connect('checkpoints.db') as conn:
        memory = AsyncSqliteSaver(conn, serde=JsonPlusSerializer(pickle_fallback=True))
        agent = CreateToolAgent().build(memory=memory)

        thread_id = tool_info['thread']
        config: RunnableConfig = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 500}

        response = await agent.aget_state(config)
        rich.print(response.interrupts)


async def continue_run(index: int, message: Optional[str] = None, *, approve: bool = False) -> None:
    """响应中断信息."""
    import aiosqlite
    import polars as pl
    import rich
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.types import Command

    from llm.agents.tool_manager import CreateToolAgent
    from llm.states import CreateToolContext

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


async def rebuild_vector_store() -> None:
    """重建工具库向量索引."""
    import polars as pl
    from langchain_core.documents import Document
    from langchain_milvus import BM25BuiltInFunction, Milvus
    from sqlmodel import Session, col, select
    from tqdm import tqdm

    from app.core.config import get_settings
    from app.core.session import engine
    from app.models.tools import ToolDefine
    from llm.models import qwen3_embedding

    embedding = qwen3_embedding()
    vectorstore = Milvus(
        collection_name='ToolCollection',
        connection_args={'uri': get_settings().knowledge_base.MILVUS_URL},
        embedding_function=embedding,
        builtin_function=BM25BuiltInFunction(),
        vector_field=['dense', 'sparse'],
        index_params=[{'index_type': 'HNSW', 'metric_type': 'L2'}, {}],
        consistency_level='Strong',
        drop_old=True,
        auto_id=True,
    )

    function_df = pl.read_parquet('tool_functions_deepseek.parquet').filter(pl.col('tool_node').is_not_null())
    node_uids = function_df['tool_node'].unique().to_list()

    with Session(engine) as session:
        statement = select(
            ToolDefine.uid,
            ToolDefine.description,
            ToolDefine.image_uid,
            ToolDefine.name,
        ).where(col(ToolDefine.uid).in_(node_uids))
        results = session.exec(statement).all()
        tools_map = {str(row.uid): row for row in results}  # type: ignore[attr-defined]

    docs_to_add = []
    empty_tools = []
    batch_size = 10

    for node_uid in tqdm(node_uids, total=len(node_uids)):
        tool_data = tools_map.get(node_uid)

        if tool_data and tool_data.description:  # type: ignore[attr-defined]
            docs_to_add.append(
                Document(
                    page_content=tool_data.description,  # type: ignore[attr-defined]
                    metadata={
                        'image_uid': str(tool_data.image_uid),  # type: ignore[attr-defined]
                        'tool_uid': str(tool_data.uid),  # type: ignore[attr-defined]
                        'name': tool_data.name,  # type: ignore[attr-defined]
                    },
                )
            )

            if len(docs_to_add) >= batch_size:
                await vectorstore.aadd_documents(docs_to_add)
                docs_to_add = []
        else:
            empty_tools.append(node_uid)

    if docs_to_add:
        await vectorstore.aadd_documents(docs_to_add)

    if empty_tools:
        empty_file = Path('empty_tools.txt')
        empty_file.write_text('\n'.join(empty_tools) + '\n')


def statistic() -> None:
    """统计生成情况."""
    import polars as pl
    import rich

    last_tool = 0
    function_df = pl.read_parquet('tool_functions_deepseek.parquet')
    print(function_df.height)
    analysed = function_df.filter(pl.col('index') > last_tool).filter(pl.col('thread').is_not_null())
    tool_results = analysed.filter(pl.col('tool_node').is_not_null())

    print(f'从上次统计, 已分析{analysed.height}条数据')
    print(f'其中成功得到有效工具节点的有: {tool_results.height}条')
    print(f'新生成节点{tool_results["tool_node"].n_unique()}个')
    print('=' * 50)
    print('最新运行记录为: ')
    rich.print(analysed.top_k(1, by='index').row(0, named=True))


def _log_update(namespace: tuple[str], mode: str, data: dict[str, Any]) -> None:
    if mode == 'custom':
        logger.info(data)
    else:
        filtered_data = {k: v for k, v in data.items() if v is not None}
        if filtered_data:
            running_node = ','.join([n.split(':')[0] for n in namespace])
            logger.debug(f'({running_node}) {filtered_data}'[:120])


@click.group()
def cli() -> None:
    """数据库初始化和管理工具."""


@cli.command()
@click.option('--drop-old', is_flag=True, default=False, help='删除旧数据 (默认: 不删除)')
def init_images_cmd(*, drop_old: bool) -> None:
    """初始化镜像数据."""
    asyncio.run(init_images(drop_old=drop_old))


@cli.command()
@click.option('--index', type=int, default=None, help='指定要处理的数据索引 (如果不指定则处理所有未完成的数据)')
def init_tools_cmd(index: Optional[int]) -> None:
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
def continue_run_cmd(index: int, message: Optional[str], *, approve: bool) -> None:
    """继续运行指定索引的工具创建任务."""
    asyncio.run(continue_run(index, message, approve=approve))


@cli.command()
@click.argument('indices', type=int, nargs=-1, required=True)
def delete_prebuild_tool_cmd(indices: tuple[int, ...]) -> None:
    """删除指定索引的预构建工具 (可一次删除多个, 例如: 61 62 63 65)."""
    asyncio.run(delete_prebuild_tool(list(indices)))


@cli.command()
def rebuild_vector_store_cmd() -> None:
    """重建向量存储."""
    asyncio.run(rebuild_vector_store())


@cli.command()
def show_statistic() -> None:
    """显示节点生成统计数据."""
    statistic()


if __name__ == '__main__':
    cli()
