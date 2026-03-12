import json
import sys
from pathlib import Path
from typing import Any, Optional

from celery import Task
from docker.types import Mount
from loguru import logger

from app.core.config import get_settings
from app.models.tools import FileMount, ImageConfig
from app.models.workflow import ToolOutput
from app.schemas.react_flow import TaskInput
from app.schemas.task_instance import TaskUpdate
from app.tasks.base.base_task import (
    BaseBioTask,
    get_tool_data,
    get_tool_output,
    update_monitor_redis,
    update_task,
)
from app.utils.input_format import PartialFormatter, build_command_template
from worker import celery_app, monitor_redis_client


def _resolve_mount_path(mount_paths: dict[str, Path], container_mount_path: str, host_mount_path: Path) -> str:
    """当不同宿主机路径被挂载到同一容器路径时, 自动分配一个不冲突的容器挂载路径.

    策略: 使用 {原容器路径}/{宿主机目录名} 作为新路径, 若仍冲突则追加序号.
    """
    abs_host = host_mount_path.absolute()
    existing = mount_paths.get(container_mount_path)

    # 无冲突: 路径未使用, 或已映射到同一宿主机目录
    if existing is None or existing == abs_host:
        return container_mount_path

    # 冲突: 用宿主机目录名做子路径
    candidate = f'{container_mount_path}/{host_mount_path.name}'
    counter = 1
    while candidate in mount_paths and mount_paths[candidate] != abs_host:
        candidate = f'{container_mount_path}/{host_mount_path.name}_{counter}'
        counter += 1

    logger.warning(f'挂载冲突: {container_mount_path} 已映射到 {existing}, 将 {abs_host} 重定向到 {candidate}')
    return candidate


def build_mounts(
    file_mounts: list[FileMount], host_output_dir: str, sample_name: str, inputs: dict[str, TaskInput]
) -> tuple[dict[str, str], dict[str, Path], ToolOutput]:
    """构建挂载映射."""
    param_inputs: dict[str, str] = {}  # 用于构建参数的文件路径
    mount_paths: dict[str, Path] = {}  # 存储挂载的路径映射
    expected_output = ToolOutput()  # 预期输出文件路径

    # 内部变量:
    # host_file_path: 宿主机上文件的路径
    # host_mount_path: 宿主机上要挂载的文件夹 (仅目录/前缀模式使用)
    # container_file_path: 容器内文件的路径
    # file.mount_path: 容器内的文件夹映射 (定义值)
    # actual_mount_path: 冲突解决后实际使用的容器挂载路径 (仅目录/前缀模式使用)
    # 注意: mount_paths 同时包含文件级挂载(key为容器内文件路径)和目录级挂载(key为容器内目录路径)

    for file in file_mounts:
        if file.file_type == 'INPUT':
            # 输入文件, 提取资源文件
            kwarg = inputs[file.name]
            file_path = get_tool_output(kwarg.source_uid, kwarg.source_k)
            host_file_path = Path(file_path)

            if host_file_path.is_file():
                # 文件: 直接挂载文件本身, 不同文件可以共存于同一容器目录
                container_file_path = f'{file.mount_path}/{host_file_path.name}'
                mount_paths[container_file_path] = host_file_path.absolute()
            elif host_file_path.is_dir():
                # 文件夹: 整目录挂载, 需要冲突检查
                host_mount_path = host_file_path
                actual_mount_path = _resolve_mount_path(mount_paths, file.mount_path, host_mount_path)
                container_file_path = actual_mount_path
                if actual_mount_path not in mount_paths:
                    mount_paths[actual_mount_path] = host_mount_path.absolute()
            else:
                # 不存在的路径, 视为文件夹+文件名前缀, 取父目录按文件夹处理
                host_mount_path = host_file_path.parent
                actual_mount_path = _resolve_mount_path(mount_paths, file.mount_path, host_mount_path)
                container_file_path = f'{actual_mount_path}/{host_file_path.name}'
                if actual_mount_path not in mount_paths:
                    mount_paths[actual_mount_path] = host_mount_path.absolute()

        else:
            # 输出文件, 添加期望输出
            try:
                host_file_path = Path(file.file_path.format(output_dir=host_output_dir, sample_name=sample_name))
            except KeyError as e:
                logger.error(file.file_path)
                logger.exception(e)
                sys.exit(1)

            host_mount_path = Path(host_output_dir)
            actual_mount_path = _resolve_mount_path(mount_paths, file.mount_path, host_mount_path)
            container_file_path = f'{actual_mount_path}/{host_file_path.name}'
            if actual_mount_path not in mount_paths:
                mount_paths[actual_mount_path] = host_mount_path.absolute()

            if file.is_log:
                expected_output.log = host_file_path.absolute()
            elif file.is_report:
                expected_output.reports[file.name] = host_file_path.absolute()
            else:
                expected_output.result[file.name] = host_file_path.absolute()

        # 添加指令组成(容器内路径)
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

    # 运行docker命令
    log_count = 0
    if get_settings().docker.CONTAINER == 'podman':
        from app.utils.container.podman import run_podman_background  # noqa: PLC0415

        for stats in run_podman_background(
            docker_image, command=tool_command, mounts=volume_binds, workdir='/data/output'
        ):
            monitor_redis_client.rpush(f'task-monitor-{self.request.id}', json.dumps(stats))
            monitor_redis_client.expire(f'task-monitor-{self.request.id}', 60 * 60 * 24)
            log_count += 1

            if log_count % 100 == 0:
                update_monitor_redis(self.request.id)
    else:
        from app.utils.container.docker import run_docker_background  # noqa: PLC0415

        for stats in run_docker_background(
            docker_image, command=tool_command, mounts=volume_binds, working_dir='/data/output'
        ):
            monitor_redis_client.rpush(f'task-monitor-{self.request.id}', json.dumps(stats))
            monitor_redis_client.expire(f'task-monitor-{self.request.id}', 60 * 60 * 24)
            log_count += 1

            if log_count % 100 == 0:
                update_monitor_redis(self.request.id)

    # 验证输出文件是否生成
    missing_outputs = [str(path) for name, path in expected_output.all_files().items() if not path.exists()]
    if missing_outputs:
        logger.warning(f'部分输出文件未生成: {missing_outputs}')

    # 返回结果
    return expected_output
