from typing import Literal

import matplotlib.pyplot as plt
import polars as pl
import rich
import seaborn as sns
from loguru import logger


def load_run_data() -> pl.DataFrame:
    """读取节点生成记录."""
    df_tool = pl.read_parquet('tool_functions_deepseek.parquet')
    df_run = pl.read_parquet('test_run.parquet')

    tasks = (
        df_tool.filter(pl.col('thread').is_not_null())
        .filter(~pl.col('interrupted'))
        .select(
            [
                pl.col('thread').alias('test_run_id'),
                pl.col('function_description').alias('task'),
                pl.col('tool_node'),
            ]
        )
        .unique(subset=['test_run_id'], keep='first')
    )

    run_summary = (
        df_run.with_columns(
            pl.col('messages').list.len().fill_null(0).alias('__msg_len')  # 用临时列名
        )
        .sort('test_run_id')
        .group_by('test_run_id', maintain_order=True)
        .agg(
            [
                pl.col('answer').filter(pl.col('agent') == 'test_run').first().alias('answer'),
                pl.col('__msg_len').sum().cast(pl.Int64).alias('message_len'),
            ]
        )
        .select(['test_run_id', 'answer', 'message_len'])
    )

    out = tasks.join(run_summary, on='test_run_id', how='inner').select(
        ['test_run_id', 'task', 'answer', 'tool_node', 'message_len']
    )

    logger.info(f'共计{out.height}条运行记录')
    logger.info(f'失败记录{out.filter(pl.col("tool_node").is_null()).height}条')
    return out


def draw_fig() -> None:
    """绘制统计图片."""
    statistic_df = load_run_data().with_columns(pl.col('tool_node').is_not_null().cast(pl.Int8).alias('success'))

    # Set Nature style
    # simple, clean, often uses Helvetica/Arial, tick marks inward, no top/right spines usually unless boxed
    sns.set_theme(style='ticks', font_scale=1.2)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0

    # Create figure with dual y-axis
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Determine bins (e.g., width of 5)
    bin_width = 5
    max_val = statistic_df['message_len'].max() if statistic_df['message_len'].max() else 0
    bins = range(0, int(max_val) + bin_width, bin_width)

    # Plot Histogram (Message Length Distribution)
    sns.histplot(
        data=statistic_df,
        x='message_len',
        bins=bins,
        ax=ax1,
        color='#3c5488',  # Nature-like blue
        alpha=0.6,
        edgecolor='white',
        label='Count',
    )
    ax1.set_xlabel('Message Length (number of turns)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold', color='#3c5488')
    ax1.tick_params(axis='y', labelcolor='#3c5488')

    # Calculate success rate per bin using Polars
    # 1. Create a bin column
    # 2. Group by bin and calculate mean of success
    success_rate_pl = (
        statistic_df.with_columns(
            ((pl.col('message_len') / bin_width).floor().cast(pl.Int64) * bin_width).alias('bin_start')
        )
        .group_by('bin_start')
        .agg(pl.col('success').mean().alias('success_rate'))
        .sort('bin_start')
        .with_columns((pl.col('bin_start') + bin_width / 2).alias('bin_center'))
    )

    # Plot Success Rate
    ax2 = ax1.twinx()
    sns.lineplot(
        data=success_rate_pl,
        x='bin_center',
        y='success_rate',
        ax=ax2,
        color='#e64b35',  # Nature-like red
        marker='o',
        linewidth=2,
        markersize=8,
        label='Success Rate',
    )
    ax2.set_ylabel('Success Rate', fontsize=12, fontweight='bold', color='#e64b35')
    ax2.tick_params(axis='y', labelcolor='#e64b35')
    ax2.set_ylim(0, 1.1)

    # Title and cleanup
    plt.title('Message Length Distribution and Success Rate', fontsize=14, pad=20)
    sns.despine(ax=ax1, right=False)  # Keep right spine for secondary axis
    ax2.spines['top'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig('imgs/message_len_distribution.png', dpi=300)
    plt.savefig('imgs/message_len_distribution.pdf')
    logger.info('Saved figures to imgs/message_len_distribution.png and imgs/message_len_distribution.pdf')


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
    result = function_df.filter(pl.col('tool_name').str.contains('Picard'))

    for row in result.iter_rows(named=True):
        # print(row['index'])
        # print(row['tool_name'])
        # print(row['function_description'])
        rich.print(row)
        print('=' * 10)


if __name__ == '__main__':
    draw_fig()
