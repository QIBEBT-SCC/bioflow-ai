import gzip
import shutil
from pathlib import Path
from typing import Any

from celery import Task
from loguru import logger

from app.models.workflow import ToolOutput
from app.schemas.react_flow import TaskInput
from app.tasks.base.base_task import BaseBioTask, get_tool_output
from worker import celery_app


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def select_file_task(
    self: Task,  # noqa: ARG001
    result_dict: dict[str, Any] | None = None,  # noqa: ARG001
    *,
    inputs: dict[str, TaskInput],
    file_name: str,
) -> ToolOutput:
    """从文件夹中提取指定文件."""
    kwarg = inputs.get('folder_path')
    parent_dir = get_tool_output(kwarg.source_uid, kwarg.source_k)

    selected_file = Path(parent_dir) / file_name

    if not Path(selected_file).exists():
        raise FileNotFoundError(parent_dir)

    return ToolOutput(result={'file_path': selected_file})


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def collect_files_task(
    self: Task,  # noqa: ARG001
    result_dict: dict[str, Any] | None = None,  # noqa: ARG001
    *,
    inputs: dict[str, TaskInput | list[TaskInput]],
    output_dir: str,
) -> ToolOutput:
    """从文件夹中提取指定文件."""
    input_files = inputs.get('files')
    kwargs = input_files if isinstance(input_files, list) else [input_files]

    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    for kwarg in kwargs:
        origin_file = Path(get_tool_output(kwarg.source_uid, kwarg.source_k))
        if origin_file.exists():
            new_file = output_folder / origin_file.name
            if new_file.exists():
                new_file.unlink()
            new_file.symlink_to(origin_file)
        else:
            logger.warning(f'file {origin_file} did not exist!')

    return ToolOutput(result={'folder_path': output_folder})


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def gzip_file_task(
    self: Task,  # noqa: ARG001
    result_dict: dict[str, Any] | None = None,  # noqa: ARG001
    *,
    inputs: dict[str, TaskInput],
) -> ToolOutput:
    """对原始文件进行压缩."""
    input_kwarg = inputs.get('file', '')
    origin_file = Path(get_tool_output(input_kwarg.source_uid, input_kwarg.source_k))
    if origin_file.exists():
        zipped_file = origin_file.with_name(f'{origin_file.name}.gz')
        with origin_file.open('rb') as f_in, gzip.open(zipped_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        return ToolOutput(result={'zipped_file': zipped_file})

    raise FileNotFoundError(origin_file)
