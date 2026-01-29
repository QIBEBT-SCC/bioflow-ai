import re
import shutil
from pathlib import Path
from typing import Literal, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from app.utils.system import format_size
from llm.states import FolderContext

# 设置特别长文件名的阈值
NAME_LENGTH_THRESHOLD = 40

# 设置文件最大字符数限制
MAX_FILE_CHARS = 4000


def _is_safe_path(path: str, root_path: str) -> bool:
    """检查路径是否在根目录范围内（安全检查）。"""
    try:
        full_path = (Path(root_path) / Path(path)).resolve()
        return full_path.is_relative_to(root_path)
    except (ValueError, RuntimeError):
        return False


def _get_safe_path(path: str, root_path: str) -> Path:
    """获取安全的绝对路径。"""
    full_path = (Path(root_path) / Path(path)).resolve()
    if not full_path.is_relative_to(root_path):
        raise ValueError(f'路径越界: {path}')
    return full_path


@tool(parse_docstring=True)
def list_directory(path: str, runtime: ToolRuntime[FolderContext]) -> str:
    """列出目录中的文件和子目录

    ## 使用案例
    - list_directory(".")  # 列出根目录
    - list_directory("src")  # 列出 src 子目录

    Args:
        path: 相对于根目录的路径
    """
    runtime.stream_writer({'data': f'list dir {path}'})
    try:
        dir_path = _get_safe_path(path, runtime.context.root_path)

        if not dir_path.exists():
            return f'错误: 路径不存在: {path}'
        if not dir_path.is_dir():
            return f'错误: 不是目录: {path}'

        # 收集所有项目信息
        entries = []
        for item in dir_path.iterdir():
            name = item.name
            if item.is_dir():
                entries.append({'name': name + '/', 'size': None, 'is_dir': True})
            else:
                size = item.stat().st_size
                size_str = format_size(size)
                entries.append({'name': name, 'size': size_str, 'is_dir': False})

        if not entries:
            return f'目录为空: {path}'

        # 计算对齐宽度（只考虑不超过阈值的文件名）
        normal_names = [e['name'] for e in entries if len(e['name']) <= NAME_LENGTH_THRESHOLD]
        max_name_len = max(len(name) for name in normal_names) if normal_names else 0
        name_width = max_name_len + 4

        # 计算size的最大宽度
        sizes = [e['size'] for e in entries if e['size'] is not None]
        max_size_len = max(len(s) for s in sizes) if sizes else 0
        size_width = max_size_len + 2

        # 格式化输出
        lines = []
        for entry in entries:
            name = entry['name']
            if entry['is_dir']:
                lines.append(name)
            else:
                size_str = entry['size']
                if len(name) > NAME_LENGTH_THRESHOLD:
                    # 特别长的文件名，只隔4个空格
                    lines.append(f'{name}    {size_str}')
                else:
                    # 普通文件名，按照统一宽度对齐
                    lines.append(f'{name:<{name_width}}{size_str:>{size_width}}')

        return '\n'.join(lines)
    except Exception as e:
        return f'错误: {str(e)}'


@tool
def read_file(
    path: str,
    runtime: ToolRuntime[FolderContext],
    offset: Optional[int] = None,
    limit: Optional[int] = None,
) -> str:
    """读取文件内容，支持行号和分页读取。

    ## 使用案例
    - read_file("config.yaml")  # 读取整个文件
    - read_file("large.log", offset=100, limit=20)  # 读取第100-120行

    Args:
        path: 相对于根目录的文件路径
        offset: 起始行号（从1开始），用于大文件分页读取
        limit: 读取的行数限制，用于大文件分页读取

    Returns:
        文件内容，每行前带有行号
    """
    runtime.stream_writer({'data': f'read file {path}'})
    try:
        file_path = _get_safe_path(path, runtime.context.root_path)

        if not file_path.exists():
            return f'错误: 文件不存在: {path}'
        if not file_path.is_file():
            return f'错误: 不是文件: {path}'

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.splitlines(keepends=True)

        total_lines = len(lines)
        start = (offset - 1) if offset else 0
        end = (start + limit) if limit else total_lines

        if start >= total_lines:
            return f'错误: offset ({offset}) 超出文件行数 ({total_lines})'

        selected_lines = lines[start:end]
        numbered_lines = [f'{i + start + 1:6d}| {line.rstrip()}' for i, line in enumerate(selected_lines)]

        result = f'文件: {path}\n'
        result += f'总行数: {total_lines}, 显示: {start + 1}-{min(end, total_lines)}\n'
        result += '-' * 50 + '\n'
        content_lines = '\n'.join(numbered_lines)

        # 检查实际返回内容的字符数，如果超过限制则截断
        header = result
        if len(header) + len(content_lines) > MAX_FILE_CHARS:
            truncated_content = content_lines[:MAX_FILE_CHARS]
            last_newline = truncated_content.rfind('\n')
            if last_newline > 0:
                truncated_content = truncated_content[:last_newline]

            result = (
                header
                + truncated_content
                + f'\n\n... [内容已截断，总计 {len(content_lines):,} 字符，已显示 {len(truncated_content):,} 字符] ...'
            )
        else:
            result = header + content_lines

        return result
    except UnicodeDecodeError:
        return f'错误: 无法解码文件 (可能是二进制文件): {path}'
    except Exception as e:
        return f'错误: {str(e)}'


@tool(parse_docstring=True)
def grep_search(
    pattern: str,
    file_path: str,
    runtime: ToolRuntime[FolderContext],
    output_mode: Literal['content', 'count'] = 'content',
    context_lines: int = 2,
) -> str:
    """
    在指定文件中搜索匹配的文本内容（支持正则表达式）。

    ## 使用案例
    - grep_search("def main", "main.py")  # 在main.py中搜索 "def main"
    - grep_search("TODO", "src/utils.py", output_mode="count")
    - grep_search("error.*exception", "app.log", context_lines=5)

    Args:
        pattern: 要搜索的正则表达式模式
        file_path: 要搜索的文件路径（相对于根目录）
        output_mode: 输出模式, "content": 显示匹配行及上下文; "count": 只显示匹配次数
        context_lines: 显示匹配行前后的上下文行数（仅content模式）

    Returns:
        搜索结果
    """
    runtime.stream_writer({'data': f'search in file {file_path} for "{pattern}"'})
    try:
        target_path = _get_safe_path(file_path, runtime.context.root_path)

        if not target_path.exists():
            return f'错误: 文件不存在: {file_path}'
        if not target_path.is_file():
            return f'错误: 不是文件: {file_path}'

        # 编译正则表达式
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f'错误: 无效的正则表达式: {e}'

        # 读取文件内容
        with open(target_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.splitlines(keepends=True)

        # 查找匹配行
        matches = []
        for i, line in enumerate(lines, 1):
            if regex.search(line):
                matches.append(i)

        if not matches:
            return f"文件 '{file_path}' 中没有找到匹配 '{pattern}' 的内容"

        # 根据输出模式返回结果
        if output_mode == 'count':
            return f"文件 '{file_path}' 中找到 {len(matches)} 个匹配"

        # content 模式：显示匹配行及上下文
        result = f"文件 '{file_path}' 中找到 {len(matches)} 个匹配:\n"
        result += '=' * 50 + '\n'

        # 逐个添加匹配项，检查是否超过字符限制
        total_matches = len(matches)
        displayed_matches = 0

        for match_line in matches:
            start = max(1, match_line - context_lines)
            end = min(len(lines), match_line + context_lines)

            match_content = f'\n行 {match_line}:\n'
            for i in range(start, end + 1):
                prefix = '> ' if i == match_line else '  '
                match_content += f'{prefix}{i:4d}| {lines[i - 1].rstrip()}\n'

            # 检查添加这个匹配项后是否超过限制
            if len(result) + len(match_content) + 200 > MAX_FILE_CHARS:
                result += f'\n... [已截断，共 {total_matches} 个匹配，已显示 {displayed_matches} 个] ...'
                break

            result += match_content
            displayed_matches += 1

        return result

    except UnicodeDecodeError:
        return f'错误: 无法解码文件 (可能是二进制文件): {file_path}'
    except Exception as e:
        return f'错误: {str(e)}'


@tool(parse_docstring=True)
def find_files(pattern: str, runtime: ToolRuntime[FolderContext], path: str = '.') -> str:
    """按文件名模式查找文件（支持通配符）。

    ## 使用案例
    - find_files("*.py")  # 查找所有Python文件
    - find_files("**/*.json", "src")  # 在src目录递归查找JSON文件
    - find_files("config.*")  # 查找所有config配置文件

    Args:
        pattern: 文件名模式，支持通配符
        path: 搜索起始路径（默认为根目录）

    Returns:
        找到的文件列表，按修改时间排序
    """
    runtime.stream_writer({'data': f'search file with "{pattern}"'})
    try:
        search_path = _get_safe_path(path, runtime.context.root_path)

        if not search_path.exists():
            return f'错误: 路径不存在: {path}'
        if not search_path.is_dir():
            return f'错误: 不是目录: {path}'

        # 使用rglob查找文件
        if '**' in pattern:
            # 递归查找
            found_files = list(search_path.glob(pattern))
        else:
            # 非递归查找
            found_files = list(search_path.glob(pattern))
            # 也包含子目录中的匹配
            found_files.extend(search_path.glob(f'**/{pattern}'))

        # 过滤出文件（排除目录）并按修改时间排序
        files = [f for f in found_files if f.is_file()]
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        if not files:
            return f"没有找到匹配 '{pattern}' 的文件"

        # 收集文件信息
        entries = []
        for file_path in files:
            rel_path = str(file_path.relative_to(runtime.context.root_path))
            size = file_path.stat().st_size
            size_str = format_size(size)
            entries.append({'name': rel_path, 'size': size_str})

        # 计算对齐宽度（只考虑不超过阈值的文件名）
        normal_names = [e['name'] for e in entries if len(e['name']) <= NAME_LENGTH_THRESHOLD]
        max_name_len = max(len(name) for name in normal_names) if normal_names else 0
        name_width = max_name_len + 4

        # 计算size的最大宽度
        sizes = [e['size'] for e in entries]
        max_size_len = max(len(s) for s in sizes) if sizes else 0
        size_width = max_size_len + 2

        # 格式化输出
        lines = []
        for entry in entries:
            name = entry['name']
            size_str = entry['size']
            if len(name) > NAME_LENGTH_THRESHOLD:
                # 特别长的文件名，只隔4个空格
                lines.append(f'{name}    {size_str}')
            else:
                # 普通文件名，按照统一宽度对齐
                lines.append(f'{name:<{name_width}}{size_str:>{size_width}}')

        return '\n'.join(lines)

    except Exception as e:
        return f'错误: {str(e)}'


@tool(parse_docstring=True)
def write_to_file(
    path: str,
    content: str,
    runtime: ToolRuntime[FolderContext],
    mode: Literal['append', 'overwrite'] = 'append',
) -> str:
    """编辑或创建文件
    注意，使用 overwrite 会完全覆盖文件中现有的内容，除非你十分确信，否则正常情况下只使用 append 模式

    ## 使用示例
    - edit_file("SUMMARY.md", "# 项目说明\\n")  # 追加内容
    - edit_file("SUMMARY.md", "新的任务列表", mode="overwrite")  # 覆盖内容

    Args:
        path: 相对于根目录的文件路径
        content: 要写入的内容
        mode: 写入模式. "append": 追加到文件末尾（默认）; "overwrite": 覆盖整个文件

    Returns:
        操作结果信息
    """
    runtime.stream_writer({'data': f'{mode} to file {path}'})
    try:
        file_path = _get_safe_path(path, runtime.context.root_path)

        # 检查文件名是否在允许列表中
        if path not in runtime.context.allowed_files:
            allowed_list = ', '.join(runtime.context.allowed_files)
            return f'错误: 不允许编辑此文件。只允许编辑以下文件: {allowed_list}'

        # 确保父目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 检查是否是新文件
        is_new_file = not file_path.exists()

        if is_new_file:
            # 新建文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f'成功: 已创建新文件 {path}'
        else:
            # 文件已存在
            if mode == 'overwrite':
                # 覆盖模式
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f'成功: 已覆盖文件 {path}'
            else:
                # 追加模式
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(content)
                return f'成功: 已追加内容到文件 {path} ({len(content)} 字符)'

    except Exception as e:
        return f'错误: {str(e)}'


@tool(parse_docstring=True)
def copy_file(
    source_path: str,
    target_path: str,
    runtime: ToolRuntime[FolderContext],
) -> str:
    """复制文件到新位置。

    Args:
        source_path: 源文件路径（相对于根目录）
        target_path: 目标文件路径（相对于根目录）

    Returns:
        操作结果信息
    """
    runtime.stream_writer({'data': f'copy {source_path} to {target_path}'})
    try:
        src_path = _get_safe_path(source_path, runtime.context.root_path)
        dst_path = _get_safe_path(target_path, runtime.context.root_path)

        if not src_path.exists():
            return f'错误: 源文件不存在: {source_path}'
        if not src_path.is_file():
            return f'错误: 源路径不是文件: {source_path}'
        if dst_path.exists():
            return f'错误: 目标文件已存在: {target_path}'

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

        size_str = format_size(dst_path.stat().st_size)
        return f'成功: 已复制文件 {source_path} -> {target_path} ({size_str})'

    except Exception as e:
        return f'错误: {str(e)}'
