import re


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
