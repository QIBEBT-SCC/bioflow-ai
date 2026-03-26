import json
import subprocess
from pathlib import Path
from typing import Any, Optional

from celery import Task
from loguru import logger

from app.models.workflow import ToolOutput
from app.schemas.react_flow import TaskInput
from app.tasks.base.base_task import BaseBioTask, get_tool_output
from worker import celery_app


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def bash_task(
    self: Task,  # noqa: ARG001
    result_dict: Optional[dict[str, Any]] = None,  # noqa: ARG001
    *,
    inputs: dict[str, TaskInput | list[TaskInput]],
    code: str,
    output_dir: str,
) -> ToolOutput:
    """直接用 subprocess 执行 bash 脚本.

    输入来自 bind_param 节点, 每个输入是 JSON 字符串 {param: file_path}.
    输出为脚本工作目录 (folder_path).
    """
    # 1. 收集所有 bind_param 输入, 解析 {param: file_path}
    raw_inputs = inputs.get('input_files', [])
    if not isinstance(raw_inputs, list):
        raw_inputs = [raw_inputs]

    param_args: dict[str, str] = {}
    for kwarg in raw_inputs:
        json_str = get_tool_output(kwarg.source_uid, kwarg.source_k)
        param_args.update(json.loads(json_str))

    # 2. 准备输出目录并写入脚本
    output_path = Path(output_dir).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    script_path = output_path / '_script.sh'
    script_path.write_text(code)

    # 3. 构建命令: bash _script.sh -param /path/to/file ...
    args = [str(script_path)]
    for param, file_path in param_args.items():
        args.extend([param, file_path])

    logger.debug(f'bash_task: bash {" ".join(args)}')

    result = subprocess.run(  # noqa: S603
        ['/usr/bin/bash', *args],
        cwd=output_path,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        err_msg = f'bash script failed (exit {result.returncode}):\n{result.stderr}'
        raise RuntimeError(err_msg)

    return ToolOutput(result={'output_folder': output_path})
