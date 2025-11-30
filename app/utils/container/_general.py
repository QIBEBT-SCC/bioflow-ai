import re
from typing import Any, Optional


def _strip_ansi_escape_sequences(text: str) -> str:
    """去除文本中的 ANSI 转义序列

    ANSI 转义序列用于终端颜色和格式控制，当命令输出被捕获时，
    这些序列会显示为乱码（如 [1;34m, [m 等）。

    Args:
        text: 包含 ANSI 转义序列的文本

    Returns:
        去除 ANSI 转义序列后的文本
    """
    # ANSI 转义序列的正则表达式模式
    # 匹配 ESC[...m 格式的序列（颜色、样式等）
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def calculate_container_stats(stats: dict[str, Any]) -> Optional[dict[str, Any]]:
    """提取容器日志中的监控内容

    从 Docker stats 输出中提取关键的监控指标，包括CPU利用率、内存利用率、
    实际内存占用、网络I/O状态、块I/O状态等。

    Args:
        stats (dict[str, Any]): Podman stats 原始输出数据

    Returns:
        dict[str, float]: 格式化后的监控指标
            - cpu_usage: CPU 使用百分比
            - mem_usage: 内存使用百分比
            - mem_used: 内存使用量 (MB)
            - io_in: 输入统计
            - io_out: 输出统计
            - uptime_seconds: 日志时间戳

    Raises:
        KeyError: 当 stats 数据结构不符合预期时抛出
        ValueError: 当数值转换失败时抛出
    """
    try:
        # CPU 相关参数
        try:
            cpu_usage = (
                stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
            )
            system_cpu_usage = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
            cpu_percent = (cpu_usage / system_cpu_usage) * 100.0 if system_cpu_usage > 0 else 0.0
        except TypeError:
            cpu_percent = 0.0

        # 内存相关参数
        try:
            mem_usage = stats['memory_stats']['usage']
            mem_limit = stats['memory_stats']['limit']
            mem_percent = (mem_usage / mem_limit) * 100.0 if mem_limit > 0 else 0.0
            mem_used_mb = mem_usage / (1024 * 1024)  # Convert to MB
        except TypeError:
            mem_percent = 0.0
            mem_used_mb = 0.0

        # IO 相关参数
        try:
            io_in = 0
            io_out = 0
            for io_stat in stats['blkio_stats']['io_service_bytes_recursive']:
                if io_stat['op'] == 'read':
                    io_in = io_stat['value']
                elif io_stat['op'] == 'write':
                    io_out = io_stat['value']
        except TypeError:
            io_in = 0
            io_out = 0

        # timestamp
        timestamp = stats['read']

        return {
            'cpu_usage': cpu_percent,
            'mem_usage': mem_percent,
            'mem_used': mem_used_mb,
            'io_in': io_in / (1024 * 1024),
            'io_out': io_out / (1024 * 1024),
            'time': timestamp,
        }
    except KeyError:
        return None
