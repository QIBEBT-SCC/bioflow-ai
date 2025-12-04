import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Optional

import aiosqlite
import click
import polars as pl
import rich
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_milvus import Milvus
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from loguru import logger
from sqlmodel import Session
from tqdm import tqdm

import app.crud.tools.image as image_crud
import app.crud.tools.tool as tool_crud
import app.crud.tools.tool_group as group_crud
import app.crud.tools.tool_tag as tag_crud
from app.core.config import get_settings
from app.core.session import engine
from app.models.tools import ImageConfig, ToolGroup, ToolImage, ToolTag
from llm.agents.tool_manager import CreateToolAgent, CreateToolContext, CreateToolState
from llm.models import qwen3_embedding


def init_tag_and_group(drop_old: bool = True):
    with Session(engine) as session:
        if drop_old:
            delete_tag_count = tag_crud.drop_all(session)
            delete_group_count = group_crud.drop_all(session)
            logger.info(f'Delete {delete_tag_count} tags and {delete_group_count} groups')

        # init tags
        with open('../presets/tags.json', 'r') as f:
            tags_data = json.loads(f.read())
            tags = [ToolTag.model_validate(tag) for tag in tags_data]
        tag_crud.insert_all(session, tags)

        # init group
        with open('../presets/groups.json', 'r') as f:
            groups_data = json.loads(f.read())
            groups = [ToolGroup.model_validate(group) for group in groups_data]
        group_crud.insert_all(session, groups)


async def init_images(drop_old: bool = True):
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

    batch_size = 100

    for i in tqdm(range(0, len(images), batch_size)):
        batch_images = images[i : i + batch_size]
        docs = [
            Document(page_content=img.description, metadata={'image_uid': str(img.uid), 'name': img.name})
            for img in batch_images
        ]

        await vectorstore.aadd_documents(docs)


async def init_tools():
    from openinference.instrumentation import TraceConfig
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    # configure the Phoenix tracer
    tracer_provider = register(
        endpoint='http://172.18.19.150:4317',
        project_name='build-tool',  # Default is 'default'
        protocol='grpc',
        batch=True,
        auto_instrument=False,  # Auto-instrument your app based on installed OI dependencies
    )
    trace_config = TraceConfig(hide_input_images=True, hide_embedding_vectors=True, base64_image_max_length=1000)
    LangChainInstrumentor(tracer_provider=tracer_provider).instrument(skip_dep_check=True, config=trace_config)

    function_df = pl.read_parquet('tool_functions_gpt5.parquet')
    todo_df = function_df.filter(~pl.col('thread_id').is_not_null()).filter(~pl.col('image_name').is_in(['Picard']))

    async with aiosqlite.connect('checkpoints.db') as conn:
        memory = AsyncSqliteSaver(conn, serde=JsonPlusSerializer(pickle_fallback=True))
        agent = CreateToolAgent().build(memory=memory)

        for row in todo_df.iter_rows(named=True):
            row: dict[str, Any]

            with Session(engine) as session:
                tool_image = image_crud.get_by_name(session, row['image_name'])
                assert tool_image is not None

            thread_id = f'tool_{row["index"]}_{row["image_name"]}'
            agent_input: CreateToolState = {
                'messages': [HumanMessage(content=f'Using `{row["name"]}` for {row["functions"]}.')],
                'image_uid': tool_image.uid,
                'image': tool_image.image,
            }
            config: RunnableConfig = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 500}
            context = CreateToolContext(auto_save=True, save_message=True, test_run_id=thread_id)
            function_df = function_df.with_columns(
                pl.when(pl.col('index') == row['index'])
                .then(pl.lit(thread_id))
                .otherwise(pl.col('thread_id'))
                .alias('thread_id')
            )
            function_df.write_parquet('tool_functions_gpt5.parquet', compression='snappy')

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
                    pl.when(pl.col('index') == row['index'])
                    .then(pl.lit(True))
                    .otherwise(pl.col('interrupt'))
                    .alias('interrupt'),
                )
                function_df.write_parquet('tool_functions_gpt5.parquet', compression='snappy')

                logger.error(f'{thread_id} interrupted')
                rich.print(response.interrupts)
                break
            else:
                function_df = function_df.with_columns(
                    pl.when(pl.col('index') == row['index'])
                    .then(pl.lit(response.values['tool_uid']))
                    .otherwise(pl.col('tool_node'))
                    .alias('tool_node'),
                )
                function_df.write_parquet('tool_functions_gpt5.parquet', compression='snappy')


async def read_interrupt(index: int):
    function_df = pl.read_parquet('tool_functions_gpt5.parquet')
    tool_info: dict[str, Any] = function_df.row(index=index, named=True)

    async with aiosqlite.connect('checkpoints.db') as conn:
        memory = AsyncSqliteSaver(conn, serde=JsonPlusSerializer(pickle_fallback=True))
        agent = CreateToolAgent().build(memory=memory)

        thread_id = tool_info['thread_id']
        config: RunnableConfig = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 500}

        response = await agent.aget_state(config)
        rich.print(response.interrupts)


async def continue_run(index: int, message: Optional[str] = None):
    from openinference.instrumentation import TraceConfig
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    # configure the Phoenix tracer
    tracer_provider = register(
        endpoint='http://172.18.19.150:4317',
        project_name='build-tool',  # Default is 'default'
        protocol='grpc',
        batch=True,
        auto_instrument=False,  # Auto-instrument your app based on installed OI dependencies
    )
    trace_config = TraceConfig(hide_input_images=True, hide_embedding_vectors=True, base64_image_max_length=1000)
    LangChainInstrumentor(tracer_provider=tracer_provider).instrument(skip_dep_check=True, config=trace_config)

    function_df = pl.read_parquet('tool_functions_gpt5.parquet')
    function_df = function_df.with_columns(
        pl.when(pl.col('index') == index).then(pl.lit(False)).otherwise(pl.col('interrupt')).alias('interrupt'),
    )
    tool_info: dict[str, Any] = function_df.row(index=index, named=True)

    async with aiosqlite.connect('checkpoints.db') as conn:
        memory = AsyncSqliteSaver(conn, serde=JsonPlusSerializer(pickle_fallback=True))
        agent = CreateToolAgent().build(memory=memory)

        thread_id = tool_info['thread_id']
        config: RunnableConfig = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 500}
        context = CreateToolContext(auto_save=True, save_message=True, test_run_id=thread_id)

        if message is None:
            input_message = None
        else:
            input_message = Command(resume=message)
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
                pl.when(pl.col('index') == index).then(pl.lit(True)).otherwise(pl.col('interrupt')).alias('interrupt'),
            )
            rich.print(response.interrupts)
        else:
            function_df = function_df.with_columns(
                pl.when(pl.col('index') == index)
                .then(pl.lit(response.values['tool_uid']))
                .otherwise(pl.col('tool_node'))
                .alias('tool_node'),
            )
        function_df.write_parquet('tool_functions_gpt5.parquet', compression='snappy')


async def delete_prebuild_tool(indices: list[int]):
    function_df = pl.read_parquet('tool_functions_gpt5.parquet')
    thread_ids_to_delete: list[str] = []

    for index in indices:
        tool_info: dict[str, Any] = function_df.row(index=index, named=True)
        logger.info(f'要删除的数据 (index={index}): {tool_info}')

        if tool_info['tool_node']:
            with Session(engine) as session:
                tool_crud.delete(session, tool_info['tool_node'])
                logger.info(f'删除了一条数据库信息 (index={index})')

        if tool_info['thread_id']:
            thread_ids_to_delete.append(tool_info['thread_id'])
            async with aiosqlite.connect('checkpoints.db') as conn:
                memory = AsyncSqliteSaver(conn, serde=JsonPlusSerializer(pickle_fallback=True))
                agent = CreateToolAgent().build(memory=memory)

                thread_id = 'tool_65_BLAST'
                config: RunnableConfig = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 500}

                response = await agent.aget_state(config, subgraphs=True)
                if (
                    response.tasks
                    and 'container_id' in response.tasks[0].state.values
                    and response.tasks[0].state.values['container_id']
                ):
                    # 删除残余容器
                    if get_settings().docker.CONTAINER == 'docker':
                        from app.utils.container.docker import remove_container

                        remove_container(response.tasks[0].state.values['container_id'])
                    else:
                        from app.utils.container.podman import remove_container

                        remove_container(response.tasks[0].state.values['container_id'])
                    logger.info('清理了残余容器')

                await memory.adelete_thread(tool_info['thread_id'])
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
        output_path = Path(get_settings().persistent.STORAGE_PATH).absolute() / 'testrun' / thread_id
        if output_path.exists():
            shutil.rmtree(output_path)

    # 批量更新 DataFrame
    indices_condition = pl.col('index').is_in(indices)
    function_df = function_df.with_columns(
        pl.when(indices_condition).then(None).otherwise(pl.col('thread_id')).alias('thread_id'),
        pl.when(indices_condition).then(None).otherwise(pl.col('tool_node')).alias('tool_node'),
        pl.when(indices_condition).then(False).otherwise(pl.col('interrupt')).alias('interrupt'),
    )
    function_df.write_parquet('tool_functions_gpt5.parquet', compression='snappy')
    logger.info(f'成功删除 {len(indices)} 个工具记录')


async def rebuild_vector_store():
    embedding = qwen3_embedding()
    vectorstore = Milvus(
        collection_name='ToolCollection',
        connection_args={'uri': get_settings().knowledge_base.MILVUS_URL},
        embedding_function=embedding,
        index_params={'index_type': 'HNSW', 'metric_type': 'L2'},
        # search_params={'M': 64, 'efConstruction': 100},
        consistency_level='Strong',
        drop_old=True,
        auto_id=True,
    )

    function_df = pl.read_parquet('tool_functions_gpt5.parquet').filter(pl.col('tool_node').is_not_null())
    for row in function_df.iter_rows(named=True):
        row: dict[str, Any]

        with Session(engine) as session:
            tool_data = tool_crud.get(session, row['tool_node'])

        docs = [
            Document(
                page_content=tool_data.description,
                metadata={
                    'image_uid': str(tool_data.image_uid),
                    'tool_uid': str(tool_data.uid),
                    'name': tool_data.name,
                },
            ),
        ]
        await vectorstore.aadd_documents(docs)


def _log_update(namespace: tuple[str], mode: str, data: dict[str, Any]):
    if mode == 'custom':
        logger.info(data)
    else:
        filtered_data = {k: v for k, v in data.items() if v is not None}
        if filtered_data:
            running_node = ','.join([n.split(':')[0] for n in namespace])
            logger.debug(f'({running_node}) {filtered_data}'[:120])


@click.group()
def cli():
    """数据库初始化和管理工具"""
    pass


@cli.command()
@click.option('--drop-old', is_flag=True, default=False, help='删除旧数据（默认：不删除）')
def init_tag_and_group_cmd(drop_old: bool):
    """初始化标签和分组"""
    init_tag_and_group(drop_old)


@cli.command()
@click.option('--drop-old', is_flag=True, default=False, help='删除旧数据（默认：不删除）')
def init_images_cmd(drop_old: bool):
    """初始化镜像数据"""
    asyncio.run(init_images(drop_old))


@cli.command()
def init_tools_cmd():
    """初始化工具"""
    asyncio.run(init_tools())


@cli.command()
@click.argument('index', type=int)
def read_interrupt_cmd(index: int):
    """读取工具中断信息"""
    asyncio.run(read_interrupt(index))


@cli.command()
@click.argument('index', type=int)
@click.option('--message', '-m', type=str, default=None, help='继续运行时发送的消息')
def continue_run_cmd(index: int, message: Optional[str]):
    """继续运行指定索引的工具创建任务"""
    asyncio.run(continue_run(index, message))


@cli.command()
@click.argument('indices', type=int, nargs=-1, required=True)
def delete_prebuild_tool_cmd(indices: tuple[int, ...]):
    """删除指定索引的预构建工具（可一次删除多个，例如: 61 62 63 65）"""
    asyncio.run(delete_prebuild_tool(list(indices)))


@cli.command()
def rebuild_vector_store_cmd():
    """重建向量存储"""
    asyncio.run(rebuild_vector_store())


if __name__ == '__main__':
    cli()
