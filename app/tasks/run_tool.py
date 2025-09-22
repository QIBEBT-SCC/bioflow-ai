import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Any

from celery import Task
from docker.types import Mount
from loguru import logger

from app.core.redis_client import monitor_redis_client
from app.models.tools import FileMount, ImageConfig
from app.schemas.task_instance import TaskUpdate
from app.tasks.base.base_task import (
    BaseBioTask,
    get_tool_output,
    get_tool_data,
    update_task,
    update_monitor_log,
    ToolOutput,
)
from app.schemas.react_flow import TaskInput
from app.utils.container.podman import run_podman_background
from app.utils.input_format import PartialFormatter, build_command_template
from worker import celery_app


def build_mounts(
    file_mounts: list[FileMount], host_output_dir: str, sample_name: str, inputs: dict[str, TaskInput]
) -> tuple[dict[str, str], dict[str, Path], ToolOutput]:
    param_inputs: dict[str, str] = {}  # 用于构建参数的文件路径
    mount_paths: dict[str, Path] = {}  # 存储挂载的路径映射
    expected_output = ToolOutput()  # 预期输出文件路径

    for file in file_mounts:
        if file.file_type == 'OUTPUT':
            # 输出文件，添加期望输出
            host_file_path = Path(file.file_path.format(output_dir=host_output_dir, sample_name=sample_name))
            host_file_dir = Path(host_output_dir)
            container_file_path = f'{file.mount_path}/{host_file_path.name}'

            if file.is_log:
                expected_output.log = host_file_path.absolute()
            elif file.is_report:
                expected_output.reports[file.name] = host_file_path.absolute()
            else:
                expected_output.result[file.name] = host_file_path.absolute()

        else:
            # 输入文件，提取资源文件
            kwarg = inputs[file.name]
            file_path = get_tool_output(kwarg.source_uid, kwarg.source_k)
            host_file_path = Path(file_path)
            if host_file_path.is_file():
                host_file_dir = host_file_path.parent
                container_file_path = f'{file.mount_path}/{host_file_path.name}'
            else:
                host_file_dir = host_file_path
                container_file_path = file.mount_path

        # 添加挂载项
        if file.mount_path not in mount_paths:
            mount_paths[file.mount_path] = host_file_dir.absolute()

        # 添加指令组成(容器内路径)
        param_inputs[file.name] = container_file_path

    return param_inputs, mount_paths, expected_output


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def docker_task(
    self: Task,
    result_dict: Optional[dict[str, Any]] = None,
    *,
    tool_uid: str,
    inputs: dict[str, TaskInput],
    output_dir: str,
    params: str = '',
    sample_name: Optional[str] = None,
) -> ToolOutput:
    """通用Docker工具执行任务

    Args:
        self: Celery任务
        result_dict: 用于占位，防止celery链式调用时报错
        tool_uid: 工具uid，用于从数据库获取具体信息
        inputs: 输入参数字典，格式为 {参数名: (uid, 类型)}
        output_dir: 输出目录
        params: 额外参数
        sample_name: 样本名称

    Returns:
        ToolOutput: 工具输出结果
    """
    tool_config = get_tool_data(tool_uid)

    # 创建输出目录
    host_output_dir = Path(output_dir).absolute()
    if tool_config.mkdir_output:
        host_output_dir.mkdir(parents=True, exist_ok=True)

    # 处理样本名
    # TODO

    # 处理文件挂载
    param_inputs, mount_paths, expected_output = build_mounts(tool_config.file_mounts, output_dir, sample_name, inputs)

    if expected_output.all_exist():
        return expected_output

    # 构建完整命令
    final_template = build_command_template(
        command_template=tool_config.command_template,
        dynamic_params=tool_config.dynamic_params,
        static_params=params,
        file_mounts=tool_config.file_mounts,
    )
    formatter = PartialFormatter()
    tool_command = formatter.format(final_template, output_dir='/data/output', sample_name=sample_name, **param_inputs)

    # 执行Docker命令
    docker_image = ImageConfig.model_validate(tool_config.image.image)
    if tool_config.use_temp_dir:
        with tempfile.TemporaryDirectory(prefix=f'task_{tool_config.name}_') as temp_dir:
            # 创建临时输出目录的挂载
            volume_binds: list[Mount] = []
            for container_path, host_path in mount_paths.items():
                # TODO 存疑
                if any(
                    file.mount_path == container_path
                    for file in tool_config.file_mounts
                    if file.mount_path != '/data/raw'
                ):
                    volume_binds.append(
                        Mount(type='bind', source=str(Path(temp_dir).absolute()), target=container_path)
                    )
                else:
                    volume_binds.append(Mount(type='bind', source=str(host_path), target=container_path))

            data = TaskUpdate(
                commands=f'podman run --rm {" ".join([f"-v {bind['Source']}:{bind['Source']}:rw" for bind in volume_binds])} {docker_image.get_image()} /bin/bash -c "{tool_command}"'
            )
            update_task(self.request.id, data)

            # 执行Docker命令
            log_count = 0
            for stats in run_podman_background(docker_image, command=tool_command, mounts=volume_binds):
                monitor_redis_client.rpush(f'task-monitor-{self.request.id}', json.dumps(stats))
                monitor_redis_client.expire(f'task-monitor-{self.request.id}', 60 * 60 * 24)
                log_count += 1

                if log_count % 100 == 0:
                    update_monitor_log(self.request.id)

            # 复制结果到最终输出目录
            for output_name, output_path in expected_output.all_files().items():
                relative_path = output_path.relative_to(host_output_dir.absolute())
                temp_path = Path(temp_dir) / relative_path

                if temp_path.exists():
                    shutil.copy(temp_path, host_output_dir)
                else:
                    logger.warning(f'临时输出文件不存在: {temp_path}')
    else:
        volume_binds = [
            Mount(type='bind', source=str(host_path), target=container_path)
            for container_path, host_path in mount_paths.items()
        ]

        data = TaskUpdate(
            commands=f'docker run --rm {" ".join([f"-v {bind['Source']}:{bind['Source']}:rw" for bind in volume_binds])} {docker_image.get_image()} /bin/bash -c "{tool_command}"'
        )
        update_task(self.request.id, data)

        # 运行docker命令
        log_count = 0
        for stats in run_podman_background(docker_image, command=tool_command, mounts=volume_binds):
            monitor_redis_client.rpush(f'task-monitor-{self.request.id}', json.dumps(stats))
            monitor_redis_client.expire(f'task-monitor-{self.request.id}', 60 * 60 * 24)
            log_count += 1

            if log_count % 100 == 0:
                update_monitor_log(self.request.id)

    # 验证输出文件是否生成
    missing_outputs = [str(path) for name, path in expected_output.all_files().items() if not path.exists()]
    if missing_outputs:
        logger.warning(f'部分输出文件未生成: {missing_outputs}')

    # 返回结果
    return expected_output
