from typing import Literal

import polars as pl
import rich


def update_function(idx: int, col_name: Literal['help_command', 'function_description'], content: str) -> None:
    """更新特定条目数据.

    Args:
        idx: 待更新的id
        col_name: 要更新的列
        content: 更新内容
    """
    function_df = pl.read_parquet('tool_functions_deepseek.parquet')

    if col_name == 'help_command':
        function_df = function_df.with_columns(
            pl.when(pl.col('index') == idx)
            .then(pl.lit(content))
            .otherwise(pl.col('help_command'))
            .alias('help_command')
        )
    else:
        function_df = function_df.with_columns(
            pl.when(pl.col('index') == idx)
            .then(pl.lit(content))
            .otherwise(pl.col('function_description'))
            .alias('function_description')
        )

    function_df.write_parquet('tool_functions_deepseek.parquet', compression='snappy')


def search_tool() -> None:
    """搜索特定工具的记录."""
    function_df = pl.read_parquet('tool_functions_deepseek.parquet')
    result = function_df.filter(pl.col('tool_name').str.contains('MrBayes'))

    for row in result.iter_rows(named=True):
        # print(row['index'])
        # print(row['tool_name'])
        # print(row['function_description'])
        rich.print(row)
        print('=' * 10)


if __name__ == '__main__':
    search_tool()
