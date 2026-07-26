import os
import re
import string
import unicodedata
from itertools import takewhile
from pathlib import Path
from typing import Optional

from app.schemas.react_flow import TaskNode
from app.schemas.tool_config import FileMount, ParamDefine


class PartialFormatter(string.Formatter):
    def get_value(self, key: str, args, kwargs) -> str:  # noqa: ANN001
        """允许缺省的模板填充."""
        try:
            return super().get_value(key, args, kwargs)
        except (KeyError, IndexError):
            return '{' + str(key) + '}'


def build_command_template(
    command_template: str,
    dynamic_params: list[ParamDefine],
    immutable_static_params: Optional[str],
    modifiable_static_params: Optional[str],
    file_mounts: list[FileMount],
) -> str:
    """构建到最后一级的指令模板字符串."""
    position_params: list[ParamDefine] = sorted(
        [param for param in dynamic_params if param.is_position], key=lambda x: x.index
    )
    dynamic_commands = [param.command for param in dynamic_params if not param.is_position]

    immutable_static_params = immutable_static_params or ''
    modifiable_static_params = modifiable_static_params or ''
    static_params = f'{immutable_static_params} {modifiable_static_params}'.strip()

    formatter = PartialFormatter()
    command_template = formatter.format(
        command_template,
        position_params=' '.join([param.command for param in position_params]),
        dynamic_params=' '.join(dynamic_commands),
        static_params=static_params,
    )
    final_template = formatter.format(
        command_template,
        **{
            file.name: (
                file.file_path.removeprefix('value:') if file.file_path.startswith('value:') else file.file_path
            )
            for file in file_mounts
        },
    )
    return ' '.join(final_template.split())


def get_path(r1: str | Path, r2: str | Path, *, output_dir: str | Path) -> tuple[Path, Path, Path, str]:
    """将输入的双端文件转换为pathlib格式.

    Args:
        r1: pair-end 文件 1
        r2: pair-end 文件 2
        output_dir: 软件输出目录
    """
    r1_file = r1 if isinstance(r1, Path) else Path(r1)
    r2_file = r2 if isinstance(r2, Path) else Path(r2)
    output_path = output_dir if isinstance(output_dir, Path) else Path(output_dir)

    sample_name = os.path.commonprefix([r1_file.name, r2_file.name]).strip('_R').strip('_')

    output_path.mkdir(parents=True, exist_ok=True)

    return r1_file, r2_file, output_path, sample_name


def get_sample_name_by_seq_input(r1: str, r2: str) -> str:
    """基于路径前缀判断样本名称."""
    n1, n2 = Path(r1).name, Path(r2).name
    common = ''.join(c[0] for c in takewhile(lambda x: x[0] == x[1], zip(n1, n2, strict=True)))
    return re.split(r'[_.]R[12]', common, flags=re.IGNORECASE)[0].rstrip('_')


def get_sample_name(first_layer: list[TaskNode]) -> str:
    """自动推断样本名称."""
    files = []
    for node in first_layer:
        if node.type == 'resource_sequence':
            return get_sample_name_by_seq_input(**node.data)
        if node.type == 'resource_file':
            files.append(node.data.get('file_path'))

    # TODO: 基于LLM的复杂处理
    return 'unknown'


def to_safe_path_segment(text: str, fallback: str = 'untitled', max_bytes: int = 255) -> str:
    """将任意字符串处理为标准UNIX路径名."""
    if text is None:
        text = ''

    s = unicodedata.normalize('NFKC', text)  # 量把全角/兼容字符收敛
    s = s.lower()  # 转小写
    s = re.sub(r'\s+', '-', s)  # 空白折叠为 -
    s = re.sub(r'[^a-z0-9\u4e00-\u9fff._-]', '-', s)  # 只保留 a-z 0-9 . _ - 中文(CJK Unified Ideographs)
    s = re.sub(r'-{2,}', '-', s)  # 合并连续的 -
    s = s.strip('._-')  # 去掉首尾的 . _ -

    # 避免特殊目录名
    if not s or s in {'.', '..'}:
        s = fallback

    # 避免以 - 开头, 防止命令行歧义
    if s.startswith('-'):
        s = '_' + s[1:]

    # 再清洗一次 fallback 场景
    s = s.strip('._-')
    if not s or s in {'.', '..'}:
        s = 'untitled'

    # 限制为 max_bytes 字节, 避免截断到非法 utf-8
    encoded = s.encode('utf-8')
    if len(encoded) > max_bytes:
        encoded = encoded[:max_bytes]
        while True:
            try:
                s = encoded.decode('utf-8')
                break
            except UnicodeDecodeError:
                encoded = encoded[:-1]

        s = s.strip('._-')
        if not s:
            s = 'untitled'

    return s
