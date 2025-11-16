import asyncio
import json
from typing import Any

import aiosqlite
import polars as pl
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_milvus import Milvus
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
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
from llm.agents.tool_manager.create_tool_agent import CreateToolAgent
from llm.models import qwen3_embedding
from llm.states import CreateToolContext, CreateToolState


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
        connection_args={'uri': get_settings().persistent.MILVUS_URL},
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
    function_df = pl.read_parquet('tool_functions_gpt5.parquet')
    todo_df = function_df.filter(~pl.col('thread_id').is_not_null())

    async with aiosqlite.connect('checkpoints.db') as conn:
        memory = AsyncSqliteSaver(conn, serde=JsonPlusSerializer(pickle_fallback=True))
        agent = CreateToolAgent(save_message=True).build(memory=memory)

        for row in todo_df.iter_rows(named=True):
            row: dict[str, Any]

            thread_id = f'tool_{row["index"]}_{row["name"]}'
            search_input: CreateToolState = {
                'messages': [HumanMessage(content=f'Using `{row["name"]}` for {row["functions"]}.')]
            }
            config: RunnableConfig = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 500}
            context = CreateToolContext(auto_save=True, test_run_id=thread_id)
            function_df = function_df.with_columns(
                pl.when(pl.col('index') == row['index'])
                .then(pl.lit(thread_id))
                .otherwise(pl.col('thread_id'))
                .alias('thread_id')
            )
            function_df.write_parquet('tool_functions_gpt5.parquet', compression='snappy')

            async for namespace, mode, data in agent.astream(
                search_input, config, context=context, stream_mode=['custom', 'updates'], subgraphs=True
            ):
                if mode == 'custom':
                    logger.info(data)
                else:
                    logger.debug(f'{namespace} {data}')

            response = await agent.aget_state(config)
            if response.interrupts:
                function_df = function_df.with_columns(
                    pl.when(pl.col('index') == row['index'])
                    .then(pl.lit(True))
                    .otherwise(pl.col('interrupt'))
                    .alias('interrupt'),
                )
            else:
                function_df = function_df.with_columns(
                    pl.when(pl.col('index') == row['index'])
                    .then(pl.lit(response.values['tool_uid']))
                    .otherwise(pl.col('tool_node'))
                    .alias('tool_node'),
                )
            function_df.write_parquet('tool_functions_gpt5.parquet', compression='snappy')

            break


async def delete_prebuild_tool(index: int):
    function_df = pl.read_parquet('tool_functions_gpt5.parquet')
    tool_info: dict[str, Any] = function_df.row(index=index, named=True)
    logger.info(f'要删除的数据: {tool_info}')

    if tool_info['tool_node']:
        with Session(engine) as session:
            tool_crud.delete(session, tool_info['tool_node'])
            logger.info('删除了一条数据库信息')

    if tool_info['thread_id']:
        async with aiosqlite.connect('checkpoints.db') as conn:
            memory = AsyncSqliteSaver(conn, serde=JsonPlusSerializer(pickle_fallback=True))
            await memory.adelete_thread(tool_info['thread_id'])
            logger.info('删除了一条智能体记录')

    function_df = function_df.with_columns(
        pl.when(pl.col('index') == index).then(None).otherwise(pl.col('thread_id')).alias('thread_id'),
        pl.when(pl.col('index') == index).then(None).otherwise(pl.col('tool_node')).alias('tool_node'),
        pl.when(pl.col('index') == index).then(False).otherwise(pl.col('interrupt')).alias('interrupt'),
    )
    function_df.write_parquet('tool_functions_gpt5.parquet', compression='snappy')


if __name__ == '__main__':
    # init_tag_and_group(True)
    # asyncio.run(init_images())
    asyncio.run(delete_prebuild_tool(1))
