import re

from pydantic import BaseModel, Field

CONTAINER_ROOT_USER = '0:0'


class ContainerMount(BaseModel):
    """One host path mounted into a container."""

    target: str = Field(description='Target path in the container')
    source: str = Field(description='Source path on the host')
    read_only: bool


def _strip_ansi_escape_sequences(text: str) -> str:
    """Remove ANSI escape sequences from text.

    ANSI escape sequences are used for terminal color and format control. When command output is captured,
    these sequences appear as garbled characters (such as [1;34m, [m, etc.).

    Args:
        text: Text containing ANSI escape sequences

    Returns:
        Text with ANSI escape sequences removed
    """
    # ANSI 转义序列的正则表达式模式
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)
