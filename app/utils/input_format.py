import json
import os
import string
from pathlib import Path

from app.models.tools import ParamDefine, FileMount
from app.schemas.react_flow import TaskNode


class PartialFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        try:
            return super().get_value(key, args, kwargs)
        except (KeyError, IndexError):
            return '{' + str(key) + '}'


def build_command_template(
    command_template: str, dynamic_params: list[ParamDefine], static_params: str, file_mounts: list[FileMount]
) -> str:
    """构建到最后一级的指令模板字符串"""

    position_params: list[ParamDefine] = sorted(
        [param for param in dynamic_params if param.is_position], key=lambda x: x.index
    )
    dynamic_params = [param.command for param in dynamic_params if not param.is_position]

    formatter = PartialFormatter()
    command_template = formatter.format(
        command_template,
        position_params=' '.join([param.command for param in position_params]),
        dynamic_params=' '.join(dynamic_params),
        static_params=static_params,
    )
    final_template = formatter.format(command_template, **{file.name: file.file_path for file in file_mounts})

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
        if node.type == 'sequenceInputNode':
            args = json.loads(node.args)
            return get_sample_name_by_seq_input(**args)
        elif node.type == 'fileInput':
            files.append(node.args)

    # TODO 基于LLM的复杂处理
    return 'unknown'
