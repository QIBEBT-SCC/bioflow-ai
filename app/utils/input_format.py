import os
import re
import string
import unicodedata
from pathlib import Path
from typing import Optional

from app.models.tools import FileMount, ParamDefine
from app.schemas.react_flow import TaskNode


class PartialFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
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
    """构建到最后一级的指令模板字符串"""

    position_params: list[ParamDefine] = sorted(
        [param for param in dynamic_params if param.is_position], key=lambda x: x.index
    )
    dynamic_params = [param.command for param in dynamic_params if not param.is_position]

    immutable_static_params = immutable_static_params if immutable_static_params else ''
    modifiable_static_params = modifiable_static_params if modifiable_static_params else ''
    static_params = ' '.join([immutable_static_params, modifiable_static_params])

    formatter = PartialFormatter()
    command_template = formatter.format(
        command_template,
        position_params=' '.join([param.command for param in position_params]),
        dynamic_params=' '.join(dynamic_params),
        static_params=static_params,
    )
    final_template = formatter.format(command_template, **{file.name: file.file_path for file in file_mounts})
    final_template = ' '.join(final_template.split())
    return final_template


def get_path(r1: str | Path, r2: str | Path, *, output_dir: str | Path) -> tuple[Path, Path, Path, str]:
    """将输入的双端文件转换为pathlib格式

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


def get_sample_name_by_seq_input(r1: str, r2: str):
    r1_file = Path(r1)
    r2_file = Path(r2)
    sample_name = os.path.commonprefix([r1_file.name, r2_file.name]).strip('_R').strip('_r').strip('_')

    return sample_name


def get_sample_name(first_layer: list[TaskNode]) -> str:
    files = []
    for node in first_layer:
        if node.type == 'resource_sequence':
            return get_sample_name_by_seq_input(**node.args)
        elif node.type == 'resource_file':
            files.append(node.args)

    # TODO 基于LLM的复杂处理
    return 'unknown'


def to_safe_path_segment(text: str, fallback: str = 'untitled', max_bytes: int = 255) -> str:
    """将任意字符串处理为标准UNIX路径名."""
    if text is None:
        text = ''

    s = unicodedata.normalize('NFKC', str(text))  # 量把全角/兼容字符收敛
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
