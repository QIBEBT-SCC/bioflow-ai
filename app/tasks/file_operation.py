import gzip
import json
import shutil
from pathlib import Path

from celery import Task
from loguru import logger

from app.models.workflow import ToolOutput
from app.schemas.react_flow import TaskInput
from app.tasks.base.base_task import BaseBioTask, get_prefix, get_tool_output
from app.utils.input_format import to_safe_path_segment
from script.worker import celery_app


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def select_file_task(
    self: Task,  # noqa: ARG001
    result_dict: ToolOutput | None = None,  # noqa: ARG001
    *,
    inputs: dict[str, TaskInput],
    file_name: str,
) -> ToolOutput:
    """从文件夹中提取指定文件."""
    kwarg = inputs.get('folder_path')
    if kwarg is None:
        raise ValueError
    parent_dir, sample_name = get_tool_output(kwarg.source_uid, kwarg.source_k)

    selected_file = Path(parent_dir) / file_name

    if not Path(selected_file).exists():
        raise FileNotFoundError(parent_dir)

    return ToolOutput(result={'file_path': selected_file}, sample_name=sample_name)


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def collect_files_task(
    self: Task,  # noqa: ARG001
    result_dict: ToolOutput | None = None,  # noqa: ARG001
    *,
    inputs: dict[str, TaskInput | list[TaskInput]],
    output_dir: str,
) -> ToolOutput:
    """从文件夹中提取指定文件."""
    input_files = inputs.get('files', [])
    kwargs = input_files if isinstance(input_files, list) else [input_files]

    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    for kwarg in kwargs:
        origin_file = Path(get_tool_output(kwarg.source_uid, kwarg.source_k)[0])
        if origin_file.exists():
            new_file = output_folder / origin_file.name
            if new_file.exists():
                new_file.unlink()
            new_file.symlink_to(origin_file)
        else:
            logger.warning(f'file {origin_file} did not exist!')

    return ToolOutput(result={'folder_path': output_folder})


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def collect_mount_dirs_task(
    self: Task,  # noqa: ARG001
    result_dict: ToolOutput | None = None,  # noqa: ARG001
    *,
    inputs: dict[str, TaskInput | list[TaskInput]],
    output_dir: str,
) -> ToolOutput:
    """Collect multiple upstream directories into a mount-list manifest.

    The returned value is intentionally not a normal path. Downstream tool
    execution can recognize the ``mountlist:`` prefix and expand the manifest
    into multiple container mounts under one logical input directory.
    """
    input_dirs = inputs.get('dirs', [])
    kwargs = input_dirs if isinstance(input_dirs, list) else [input_dirs]
    if not kwargs:
        err_msg = 'collect_mount_dirs requires at least one input directory'
        raise ValueError(err_msg)

    output_folder = Path(output_dir).absolute()
    output_folder.mkdir(parents=True, exist_ok=True)
    mount_root = output_folder / 'mount_root'
    mount_root.mkdir(parents=True, exist_ok=True)

    entries = []
    sample_names = []
    for index, kwarg in enumerate(kwargs, start=1):
        source_path_str, sample_name = get_tool_output(kwarg.source_uid, kwarg.source_k)
        source_path = Path(source_path_str).absolute()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if not source_path.is_dir():
            raise NotADirectoryError(source_path)

        source_name = to_safe_path_segment(source_path.name, fallback='dir')
        mount_name = f'{index:03d}_{source_name}'
        entries.append(
            {
                'mount_name': mount_name,
                'source_path': str(source_path),
                'source_uid': kwarg.source_uid,
                'source_key': kwarg.source_k,
                'sample_name': sample_name,
            }
        )
        sample_names.append(sample_name)

    manifest_path = output_folder / 'mount_manifest.json'
    manifest = {
        'version': 1,
        'type': 'mount_dir_list',
        'root_dir': str(mount_root),
        'entries': entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    common_sample_name = get_prefix(sample_names)
    return ToolOutput(result={'mount_dir_list': f'mountlist:{manifest_path}'}, sample_name=common_sample_name)


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def gzip_file_task(
    self: Task,  # noqa: ARG001
    result_dict: ToolOutput | None = None,  # noqa: ARG001
    *,
    inputs: dict[str, TaskInput],
) -> ToolOutput:
    """对原始文件进行压缩."""
    input_kwarg = inputs.get('file')
    if input_kwarg is None:
        raise ValueError
    file_path, sample_name = get_tool_output(input_kwarg.source_uid, input_kwarg.source_k)
    origin_file = Path(file_path)
    if origin_file.exists():
        zipped_file = origin_file.with_name(f'{origin_file.name}.gz')
        with origin_file.open('rb') as f_in, gzip.open(zipped_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        return ToolOutput(result={'zipped_file': zipped_file}, sample_name=sample_name)

    raise FileNotFoundError(origin_file)


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def bind_param_task(
    self: Task,  # noqa: ARG001
    result_dict: ToolOutput | None = None,  # noqa: ARG001
    *,
    inputs: dict[str, TaskInput],
    parameter: str,
) -> ToolOutput:
    """将文件绑定到指定参数名, 输出为 JSON 字符串."""
    kwarg = inputs.get('file')
    if kwarg is None:
        raise ValueError
    file_path, sample_name = get_tool_output(kwarg.source_uid, kwarg.source_k)
    return ToolOutput(result={'file_with_param': json.dumps({parameter: file_path})}, sample_name=sample_name)
