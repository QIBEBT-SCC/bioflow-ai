import asyncio

import aiosqlite
import polars as pl
import rich
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from llm.agents.tool_manager import CreateToolAgent


def read_origin_data():
    origin_df = pl.read_parquet('test_run.parquet')
    print(origin_df)
    # print(origin_df.filter(pl.col('test_run_id') == 'tool_52_bedtools').select('question').item())


def update_function():
    function_df = pl.read_parquet('tool_functions_gpt5.parquet')
    print(function_df)

    # 将索引84-93的image_name设置为GATK
    function_df = function_df.with_columns(
        pl.when(pl.col('index').is_between(84, 93))
        .then(pl.lit('GATK'))
        .otherwise(pl.col('image_name'))
        .alias('image_name')
    )
    print('\n索引84-93的数据：')
    print(function_df.filter(pl.col('index').is_between(84, 93)).select(['index', 'name', 'image_name']))

    function_df.write_parquet('tool_functions_gpt5.parquet', compression='snappy')


async def read_ckpt():
    async with aiosqlite.connect('checkpoints.db') as conn:
        memory = AsyncSqliteSaver(conn, serde=JsonPlusSerializer(pickle_fallback=True))
        agent = CreateToolAgent().build(memory=memory)

        config: RunnableConfig = {'configurable': {'thread_id': 'tool_86_GATK'}}

        result = await agent.aget_state(config, subgraphs=True)
        rich.print(result)


if __name__ == '__main__':
    asyncio.run(read_ckpt())
