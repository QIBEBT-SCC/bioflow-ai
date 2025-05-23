import shutil
import tempfile
from pathlib import Path
from typing import Optional, Any

from celery import Task
from loguru import logger

from app.models.workflow import TaskUpdate
from app.tasks.base.base_task import BaseBioTask, get_tool_output, get_tool_data, update_task
from app.models.react_flow import TaskInput
from app.models.tools import ParamType
from app.utils.docker import run_container
from worker import celery_app


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
) -> dict[str, str]:
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
    host_output_path = Path(output_dir)
    if tool_config.mkdir_output:
        host_output_path.mkdir(parents=True, exist_ok=True)

    input_cmds = []
    output_cmds = []
    mount_paths = {}  # 存储挂载路径映射
    expected_outputs = {}  # 预期输出文件路径
    param_outputs = {}  # 用于构建参数的输出文件
    container_output_dir = '/data/output'

    # 处理样本名
    # TODO

    # 处理输出文件和挂载
    for output_file in tool_config.output_files:
        # 添加挂载项
        container_output_dir = output_file.mount_path
        if container_output_dir and container_output_dir not in mount_paths:
            mount_paths[container_output_dir] = host_output_path.absolute()

        # 计算实际输出文件路径
        host_file_path = Path(
            output_file.file_path.format(output_dir=host_output_path, name=sample_name)
        )
        expected_outputs[output_file.name] = host_file_path.absolute()
        param_outputs[output_file.name] = output_file.file_path.format(
            output_dir=container_output_dir, name=sample_name
        )

    if all(path.exists() for path in expected_outputs.values()):
        return expected_outputs

    # 处理必需参数
    for param in tool_config.required_params:
        # 处理输入参数
        if param.param_type == ParamType.INPUT:
            if param.name in inputs:
                kwarg = inputs[param.name]
                file_path = get_tool_output(kwarg.source_uid, kwarg.source_k)
                input_file = Path(file_path)

                # 添加挂载项
                host_dir = input_file.parent if param.is_file else input_file
                container_dir = param.mount_path
                if container_dir and container_dir not in mount_paths:
                    mount_paths[container_dir] = host_dir

                # 构建命令
                container_file_path = f'{container_dir}/{input_file.name}'
                input_cmd = param.command.format(**{param.name: container_file_path})
                input_cmds.append(input_cmd)
            elif param.default and param.required:
                input_cmd = param.command.format(**{param.name: param.default})
                input_cmds.append(input_cmd)
            elif param.required:
                raise ValueError(f'缺少必需参数: {param.name}')

        # 处理输出参数
        elif param.param_type == ParamType.OUTPUT:
            # 构建命令
            if param.name in param_outputs:
                output_cmd = param.command.format(**{param.name: param_outputs[param.name]})
            else:
                output_cmd = param.command.format(output_dir=container_output_dir, name=sample_name)

            output_cmds.append(output_cmd)

    # 构建完整命令
    tool_command = tool_config.command_template.format(
        required_params=' '.join(input_cmds + output_cmds),
        optional_params=params,
        log=param_outputs['log'] if 'log' in param_outputs else '',
    )

    # 执行Docker命令
    if tool_config.use_temp_dir:
        with tempfile.TemporaryDirectory(prefix=f'task_{tool_config.name}_') as temp_dir:
            # 创建临时输出目录的挂载
            volume_binds = []
            for container_path, host_path in mount_paths.items():
                if any(
                    output_file.mount_path == container_path
                    for output_file in tool_config.output_files
                    if output_file.mount_path != '/data/raw'
                ):
                    volume_binds.append(f'{Path(temp_dir)}:{container_path}:rw')
                else:
                    volume_binds.append(f'{host_path}:{container_path}:rw')

            data = TaskUpdate(
                commands=f'docker run --rm {" ".join([f"-v {bind}" for bind in volume_binds])} {tool_config.docker_config.image.get_image()} /bin/bash -c "{tool_command}"'
            )
            update_task(self.request.id, data)

            # 执行Docker命令
            run_container(
                tool_config.docker_config.image,
                commands=['/bin/bash', '-c', tool_command],
                volumes_binds=volume_binds,
                **tool_config.docker_config.docker_settings,
            )

            # 复制结果到最终输出目录
            for output_name, output_path in expected_outputs.items():
                relative_path = output_path.relative_to(host_output_path.absolute())
                temp_path = Path(temp_dir) / relative_path

                if temp_path.exists():
                    shutil.copy(temp_path, host_output_path)
                else:
                    logger.warning(f'临时输出文件不存在: {temp_path}')
    else:
        volume_binds = [
            f'{host_path}:{container_path}:rw' for container_path, host_path in mount_paths.items()
        ]

        data = TaskUpdate(
            commands=f'docker run --rm {" ".join([f"-v {bind}" for bind in volume_binds])} {tool_config.docker_config.image.get_image()} /bin/bash -c "{tool_command}"'
        )
        update_task(self.request.id, data)

        # 直接使用最终输出目录
        run_container(
            tool_config.docker_config.image,
            commands=['/bin/bash', '-c', tool_command],
            volumes_binds=volume_binds,
            **tool_config.docker_config.docker_settings,
        )

    # 验证输出文件是否生成
    missing_outputs = [str(path) for name, path in expected_outputs.items() if not path.exists()]
    if missing_outputs:
        logger.warning(f'部分输出文件未生成: {missing_outputs}')

    # 返回结果
    return expected_outputs
