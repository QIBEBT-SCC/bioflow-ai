from typing import Literal

import duckdb
import matplotlib.pyplot as plt
import polars as pl
import rich
import seaborn as sns
from loguru import logger
from matplotlib.ticker import FuncFormatter


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
    print(statistic_df.filter(pl.col('success') == 0))

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

    # Determine bins (e.g., width of 10 to reduce sparsity at tail)
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

    # Prepare data for Line Plot with CI
    # We assign each point to a bin center, then let seaborn aggregate and calculate CI
    df_for_line = statistic_df.with_columns(
        ((pl.col('message_len') / bin_width).floor().cast(pl.Int64) * bin_width + bin_width / 2).alias('bin_center')
    )

    # Plot Success Rate with Confidence Interval
    ax2 = ax1.twinx()
    sns.lineplot(
        data=df_for_line,
        x='bin_center',
        y='success',
        ax=ax2,
        color='#00A087',  # Nature-like green
        marker='o',
        linewidth=2,
        markersize=6,
        err_style='band',  # Default, draws CI
        label='Success Rate',
    )
    ax2.set_ylabel('Success Rate', fontsize=12, fontweight='bold', color='#00A087')
    ax2.tick_params(axis='y', labelcolor='#00A087')
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


def draw_bidirectional_fig() -> None:
    """绘制双向直方图 (Success vs Fail)."""
    statistic_df = load_run_data().with_columns(pl.col('tool_node').is_not_null().cast(pl.Int8).alias('success'))

    # 区分成功和失败的数据
    success_df = statistic_df.filter(pl.col('success') == 1)
    fail_df = statistic_df.filter(pl.col('success') == 0)

    # Binning
    bin_width = 5
    max_val = statistic_df['message_len'].max() if statistic_df['message_len'].max() else 0
    bins = range(0, int(max_val) + bin_width, bin_width)

    # Plot
    sns.set_theme(style='ticks', font_scale=1.2)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Success (Positive Y)
    sns.histplot(
        data=success_df,
        x='message_len',
        color='#4DBBD5',  # Nature Blue
        bins=bins,
        alpha=0.7,
        label='Success',
        edgecolor='white',
        ax=ax,
    )

    # Fail (Negative Y)
    # Seaborn histplot doesn't directly support negative weights easily for simple counts in this way without workaround
    # Workaround: Calculate histogram counts first, then plot barplot with negative values OR use weights=-1
    # Using weights=-1 with histplot:
    # We need to add a weights column to fail_df
    fail_df_weighted = fail_df.with_columns(pl.lit(-1).alias('weight'))

    sns.histplot(
        data=fail_df_weighted,
        x='message_len',
        bins=bins,
        weights='weight',
        color='#E64B35',  # Nature Red
        alpha=0.7,
        label='Fail',
        edgecolor='white',
        ax=ax,
    )

    # Format Axis
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Message Length', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')

    # Fix Y-axis labels to be positive numbers
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: str(abs(int(x)))))

    plt.legend()
    plt.title('Success vs Failure Distribution by Message Length', fontsize=14, pad=20)
    sns.despine()

    plt.tight_layout()
    plt.savefig('imgs/success_fail_distribution.png', dpi=300)
    plt.savefig('imgs/success_fail_distribution.pdf')
    logger.info('Saved figures to imgs/success_fail_distribution.png and imgs/success_fail_distribution.pdf')


def load_score_data() -> pl.DataFrame:
    """读取评估打分数据."""
    conn = duckdb.connect('result.db')
    score_df = conn.execute('SELECT * FROM tool_generate').pl()
    score_df = score_df.filter(pl.col('bioflow_ai_score').is_not_null()).select(
        [
            'run_id',
            'bioflow_ai_score',
            'biomni_score',
            'biomaster_score',
            'gemini_score',
        ]
    )
    conn.close()

    return score_df


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
    # draw_fig()
    # draw_bidirectional_fig()
    load_score_data()
