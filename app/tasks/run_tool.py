import json
import sys
from pathlib import Path
from typing import Any, Optional

from celery import Task
from docker.types import Mount
from loguru import logger

from app.models.tools import FileMount, ImageConfig
from app.models.workflow import ToolOutput
from app.schemas.react_flow import TaskInput
from app.schemas.task_instance import TaskUpdate
from app.tasks.base.base_task import (
    BaseBioTask,
    get_tool_data,
    get_tool_output,
    update_task,
)
from app.utils.container import run_command_background
from app.utils.input_format import PartialFormatter, build_command_template
from worker import celery_app, monitor_redis_client


def _resolve_mount_path(
    mount_paths: dict[str, Path],
    param_inputs: dict[str, str],
    conflicted_paths: set[str],
    container_mount_path: str,
    host_mount_path: Path,
) -> str:
    """当不同宿主机目录被挂载到同一容器路径时, 自动分配不冲突的容器挂载路径.

    首次冲突时回溯: 把先占据该路径的挂载也挪到子目录, 并更新 param_inputs 中的引用.
    """
    abs_host = host_mount_path.absolute()
    existing = mount_paths.get(container_mount_path)

    # 1. 无冲突: 路径未使用且未曾冲突
    if existing is None and container_mount_path not in conflicted_paths:
        return container_mount_path

    # 2. 已映射到同一宿主机目录
    if existing == abs_host:
        return container_mount_path

    # 3. 首次冲突: 把先来的也挪到子目录
    if existing is not None and container_mount_path not in conflicted_paths:
        conflicted_paths.add(container_mount_path)
        old_host = mount_paths.pop(container_mount_path)
        old_sub = f'{container_mount_path}/{old_host.name}'
        mount_paths[old_sub] = old_host
        # 回溯更新 param_inputs 中所有引用了旧路径的条目
        for k, v in param_inputs.items():
            if v == container_mount_path or v.startswith(container_mount_path + '/'):
                relative = v[len(container_mount_path) :]
                param_inputs[k] = old_sub + relative

    # 4. 为当前挂载分配子目录
    candidate = f'{container_mount_path}/{host_mount_path.name}'
    counter = 1
    while candidate in mount_paths and mount_paths[candidate] != abs_host:
        candidate = f'{container_mount_path}/{host_mount_path.name}_{counter}'
        counter += 1

    logger.warning(f'挂载冲突: {container_mount_path} 已分裂, 将 {abs_host} 映射到 {candidate}')
    return candidate


def build_mounts(
    file_mounts: list[FileMount], host_output_dir: str, sample_name: str, inputs: dict[str, TaskInput]
) -> tuple[dict[str, str], dict[str, Path], ToolOutput]:
    """构建挂载映射."""
    param_inputs: dict[str, str] = {}  # 用于构建参数的文件路径
    mount_paths: dict[str, Path] = {}  # 存储挂载的路径映射 (始终目录级)
    expected_output = ToolOutput()  # 预期输出文件路径
    conflicted_paths: set[str] = set()  # 已发生过冲突的容器路径

    for file in file_mounts:
        if file.file_path.startswith('value:'):
            kwarg = inputs[file.name]
            value = get_tool_output(kwarg.source_uid, kwarg.source_k)
            param_inputs[file.name] = value

        else:
            if file.file_type == 'INPUT':
                kwarg = inputs[file.name]
                file_path = get_tool_output(kwarg.source_uid, kwarg.source_k)
                host_file_path = Path(file_path)

                if host_file_path.is_dir():
                    # 文件夹: 整目录挂载
                    host_mount_path = host_file_path
                    actual = _resolve_mount_path(
                        mount_paths, param_inputs, conflicted_paths, file.mount_path, host_mount_path
                    )
                    container_file_path = actual
                else:
                    # 文件或前缀: 挂载父目录, 保留伴生文件
                    host_mount_path = host_file_path.parent
                    actual = _resolve_mount_path(
                        mount_paths, param_inputs, conflicted_paths, file.mount_path, host_mount_path
                    )
                    container_file_path = f'{actual}/{host_file_path.name}'

                if actual not in mount_paths:
                    mount_paths[actual] = host_mount_path.absolute()

            else:
                # 输出文件
                try:
                    host_file_path = Path(file.file_path.format(output_dir=host_output_dir, sample_name=sample_name))
                except KeyError as e:
                    logger.error(file.file_path)
                    logger.exception(e)
                    sys.exit(1)

                host_mount_path = Path(host_output_dir)
                actual = _resolve_mount_path(
                    mount_paths, param_inputs, conflicted_paths, file.mount_path, host_mount_path
                )
                container_file_path = f'{actual}/{host_file_path.name}'
                if actual not in mount_paths:
                    mount_paths[actual] = host_mount_path.absolute()

                if file.is_log:
                    expected_output.log = host_file_path.absolute()
                elif file.is_report:
                    expected_output.reports[file.name] = host_file_path.absolute()
                else:
                    expected_output.result[file.name] = host_file_path.absolute()

            param_inputs[file.name] = container_file_path

    return param_inputs, mount_paths, expected_output


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def docker_task(
    self: Task,
    result_dict: Optional[dict[str, Any]] = None,  # noqa: ARG001
    *,
    tool_uid: str,
    inputs: dict[str, TaskInput],
    output_dir: str,
    sample_name: str,
    params: str = '',
) -> ToolOutput:
    """通用Docker工具执行任务.

    Args:
        self: Celery任务
        result_dict: 用于占位, 防止celery链式调用时报错
        tool_uid: 工具uid, 用于从数据库获取具体信息
        inputs: 输入参数字典, 格式为 {参数名: (uid, 类型)}
        output_dir: 输出目录
        params: 额外参数
        sample_name: 样本名称

    Returns:
        ToolOutput: 工具输出结果
    """
    assert self.request.id is not None  # noqa: S101

    tool_config = get_tool_data(tool_uid)
    logger.debug(f'task {self.request.id} start: {tool_config.name}')

    # 创建输出目录
    host_output_dir = Path(output_dir).absolute()
    host_output_dir.mkdir(parents=True, exist_ok=True)

    # 处理文件挂载
    param_inputs, mount_paths, expected_output = build_mounts(tool_config.file_mounts, output_dir, sample_name, inputs)

    if expected_output.all_exist():
        return expected_output

    # 构建完整命令
    final_template = build_command_template(
        command_template=tool_config.command_template,
        dynamic_params=tool_config.dynamic_params,
        immutable_static_params=tool_config.immutable_static_params,
        modifiable_static_params=params,
        file_mounts=tool_config.file_mounts,
    )
    formatter = PartialFormatter()
    tool_command = formatter.format(final_template, output_dir='/data/output', sample_name=sample_name, **param_inputs)

    # 组合docker命令
    docker_image = ImageConfig.model_validate(tool_config.image.image)
    volume_binds = [
        Mount(type='bind', source=str(host_path), target=container_path)
        for container_path, host_path in mount_paths.items()
    ]

    data = TaskUpdate(
        commands=f'docker run --rm {" ".join([f"-v {bind['Source']}:{bind['Target']}:rw" for bind in volume_binds])} {docker_image.get_image()} /bin/bash -c "{tool_command}"'
    )
    update_task(self.request.id, data)

    # 运行容器命令并采集监控数据
    redis_key = f'task-monitor-{self.request.id}'
    for stats in run_command_background(docker_image, tool_command, volume_binds, working_dir='/data/output'):
        monitor_redis_client.rpush(redis_key, json.dumps(stats))
    monitor_redis_client.expire(redis_key, 60 * 60 * 24)

    # 验证输出文件是否生成
    missing_outputs = [str(path) for name, path in expected_output.all_files().items() if not path.exists()]
    if missing_outputs:
        logger.warning(f'部分输出文件未生成: {missing_outputs}')

    # 返回结果
    return expected_output
