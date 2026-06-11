from pathlib import Path
from typing import Any

import questionary
from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# 回答完成后, 将上一行行首的问号 (?) 覆写为绿色对勾
DONE_MARK = '\x1b[1A\r\x1b[38;2;16;185;129m✓\x1b[0m\x1b[1B\r'
PROMPT_STYLE = questionary.Style(
    [
        ('qmark', 'fg:#22d3ee bold'),
        ('question', 'fg:#e5e7eb bold'),
        ('answer', 'fg:#10b981 bold'),
        ('pointer', 'fg:#f59e0b bold'),
        ('highlighted', 'fg:#22d3ee bold'),
        ('selected', 'fg:#10b981'),
        ('separator', 'fg:#64748b'),
        ('instruction', 'fg:#94a3b8'),
        ('text', 'fg:#e5e7eb'),
        ('disabled', 'fg:#64748b italic'),
    ]
)


def finish(question: questionary.Question) -> Any:
    """Submit a question and replace the leading question mark with a check mark."""
    answer = question.unsafe_ask()
    console.file.write(DONE_MARK)
    console.file.flush()
    return answer


def prompt_text(message: str, *, default: str = '', validate: Any = None) -> str:
    """Prompt for text input with the shared style."""
    return finish(questionary.text(message, default=default, validate=validate, style=PROMPT_STYLE))


def prompt_password(message: str) -> str:
    """Prompt for password input with the shared style."""
    return finish(questionary.password(message, style=PROMPT_STYLE))


def prompt_confirm(message: str, *, default: bool = False) -> bool:
    """Prompt for confirmation with the shared style."""
    return finish(questionary.confirm(message, default=default, style=PROMPT_STYLE))


def prompt_select(message: str, *, choices: list[str], default: str) -> str:
    """Prompt for a single selection with the shared style."""
    return finish(questionary.select(message, choices=choices, default=default, style=PROMPT_STYLE))


def prompt_path(message: str, *, default: str) -> str:
    """Prompt for a path with the shared style."""
    return finish(questionary.path(message, default=default, style=PROMPT_STYLE))


def render_hero() -> None:
    """Render the initialization wizard hero."""
    title = Text('BioFlowAI', style='bold #22d3ee')
    title.append(' 配置初始化向导', style='bold #f8fafc')

    body = Table.grid(padding=(0, 1))
    body.add_column(justify='center')
    body.add_row(title)
    body.add_row(Text('按步骤生成 config.toml, 默认值可直接回车确认。', style='#94a3b8'))

    console.print()
    console.print(
        Panel(
            Align.center(body),
            border_style='#22d3ee',
            box=box.ROUNDED,
            padding=(1, 4),
        )
    )
    console.print()


def render_section(title: str, description: str) -> None:
    """Render a section heading."""
    body = Table.grid(padding=(0, 1))
    body.add_column()
    body.add_row(Text(title, style='bold #f8fafc'))
    body.add_row(Text(description, style='#94a3b8'))

    console.print()
    console.print(
        Panel(
            body,
            border_style='#334155',
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def render_notice(message: str, *, style: str = '#f59e0b') -> None:
    """Render a notice message."""
    console.print(Panel(Text(message, style=style), border_style=style, box=box.ROUNDED, padding=(0, 2)))


def render_success(config_path: Path) -> None:
    """Render config initialization success."""
    body = Table.grid(padding=(0, 1))
    body.add_column()
    body.add_row(Text('配置初始化完成', style='bold #10b981'))
    body.add_row(Text(f'已写入 {config_path}', style='#cbd5e1'))

    console.print()
    console.print(
        Panel(
            body,
            title='[bold #10b981]Done[/bold #10b981]',
            border_style='#10b981',
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def render_system_success() -> None:
    """Render system initialization success."""
    console.print()
    console.print(
        Panel(
            Text('系统初始化流程完成。', style='bold #10b981'),
            title='[bold #10b981]Done[/bold #10b981]',
            border_style='#10b981',
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
