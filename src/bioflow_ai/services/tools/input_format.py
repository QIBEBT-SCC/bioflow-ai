import os
import re
import string
from itertools import takewhile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from bioflow_ai.schemas.react_flow import TaskNode
from bioflow_ai.schemas.tool_config import FileMount, ParamDefine


class _UnresolvedField(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    conversion: str | None = None


class PartialFormatter(string.Formatter):
    """Preserve runtime fields until input paths are available for safe property lookup."""

    def __init__(self, *, input_paths: dict[str, Path | None] | None = None, strict: bool = False) -> None:
        self.input_paths = input_paths
        self.strict = strict

    def get_value(self, key: int | str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Preserve missing variables during preview, or reject them in strict mode."""
        try:
            return super().get_value(key, args, kwargs)
        except (KeyError, IndexError):
            if self.strict:
                raise
            return _UnresolvedField(name=str(key))

    def get_field(self, field_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, str | int]:
        """Resolve only the three supported properties of declared path inputs."""
        if '.' not in field_name and '[' not in field_name:
            return super().get_field(field_name, args, kwargs)
        if self.input_paths is None:
            return _UnresolvedField(name=field_name), field_name

        input_name, _, attribute = field_name.partition('.')
        if attribute not in {'name', 'stem', 'suffix'} or '[' in field_name:
            message = f'Unsupported path property in {{{field_name}}}; use .name, .stem, or .suffix.'
            raise ValueError(message)
        if input_name not in self.input_paths:
            message = f'Unknown INPUT handle in {{{field_name}}}: {input_name!r}.'
            raise ValueError(message)
        path = self.input_paths[input_name]
        if path is None:
            message = f'{{{field_name}}} requires a path INPUT; value: and mountlist: inputs are not supported.'
            raise ValueError(message)
        return {'name': path.name, 'stem': path.stem, 'suffix': path.suffix}[attribute], input_name

    def convert_field(self, value: Any, conversion: str | None) -> Any:
        """Keep conversions attached to unresolved placeholders across formatting stages."""
        if isinstance(value, _UnresolvedField):
            return _UnresolvedField(name=value.name, conversion=conversion)
        return super().convert_field(value, conversion)

    def format_field(self, value: Any, format_spec: str) -> str:
        """Keep the entire unresolved field, including its format specification."""
        if isinstance(value, _UnresolvedField):
            conversion = f'!{value.conversion}' if value.conversion else ''
            specification = f':{format_spec}' if format_spec else ''
            return f'{{{value.name}{conversion}{specification}}}'
        return super().format_field(value, format_spec)


def build_command_template(
    command_template: str,
    dynamic_params: list[ParamDefine],
    immutable_static_params: str | None,
    modifiable_static_params: str | None,
    file_mounts: list[FileMount],
) -> str:
    """构建到最后一级的指令模板字符串."""
    position_params: list[ParamDefine] = sorted(  # ty: ignore[no-matching-overload]
        [param for param in dynamic_params if param.is_position],
        key=lambda x: x.index,
    )
    optional_params = [param.command for param in dynamic_params if not param.is_position]

    immutable_static_params = immutable_static_params or ''
    modifiable_static_params = modifiable_static_params or ''
    static_params = f'{immutable_static_params} {modifiable_static_params}'.strip()
    if static_params:
        optional_params.insert(0, static_params)

    formatter = PartialFormatter()
    command_template = formatter.format(
        command_template,
        position_params=' '.join([param.command for param in position_params]),
        optional_params=' '.join(optional_params),
    )
    final_template = formatter.format(
        command_template,
        **{file.name: file.file_path.removeprefix('value:').removeprefix('passthrough:') for file in file_mounts},
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
