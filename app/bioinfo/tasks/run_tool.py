import shutil
import tempfile
from pathlib import Path
from typing import Optional

from celery import Task
from loguru import logger
from pydantic import BaseModel

from app.bioinfo.base_task import BaseBioTask
from app.models.tools import ToolData, DockerToolCreate, ParamDefine, ParamType, OutputFile
from worker import celery_app


def get_tool_output(uid: str, key: str):
    """伪方法，用于临时占位"""
    datas = {
        '8c2d6f8b-69fa-4265-9f41-aca7807343a6': {
            'file_path': '/mnt/sda/gene/yeyu/scc/250421sc_96/00_raw/2503-A31/2503-A31_1.fq.gz'
        },
        '72e050f3-9fc5-48d5-84d5-f474a5eb4a33': {
            'file_path': '/mnt/sda/gene/yeyu/scc/250421sc_96/00_raw/2503-A31/2503-A31_2.fq.gz'
        },
    }
    return datas[uid][key]


class ToolOutput(BaseModel):
    """工具输出基类"""

    tool_name: str
    command: str
    outputs: dict[str, Path]


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def docker_task(
    self: Task,
    *,
    tool_config: ToolData,
    inputs: dict[str, tuple[str, str]],
    output_dir: str,
    params: str = '',
    sample_name: Optional[str] = None,
) -> ToolOutput:
    """通用Docker工具执行任务

    Args:
        self: Celery任务
        tool_config: 工具设置信息
        inputs: 输入参数字典，格式为 {参数名: (uid, 类型)}
        output_dir: 输出目录
        params: 额外参数
        sample_name: 样本名称

    Returns:
        ToolOutput: 工具输出结果
    """
    logger.info(f'运行工具: {tool_config.name}, 任务ID: {self.request.id}')

    # 创建输出目录
    host_output_path = Path(output_dir)
    if tool_config.mkdir_output:
        host_output_path.mkdir(parents=True, exist_ok=True)

    input_cmds = []
    output_cmds = []
    mount_paths = {}  # 存储挂载路径映射
    expected_outputs = {}  # 预期输出文件路径
    container_output_dir = '/data/output'

    # 处理样本名
    # TODO

    # 处理输出文件和挂载
    for output_file in tool_config.output_files:
        # 添加挂载项
        container_output_dir = output_file.mount_path
        if container_output_dir and container_output_dir not in mount_paths:
            mount_paths[container_output_dir] = host_output_path

        # 计算实际输出文件路径
        host_file_path = Path(output_file.file_path.format(output_dir=host_output_path, name=sample_name))
        expected_outputs[output_file.name] = host_file_path


    if all(path.exists() for path in expected_outputs.values()):
        return ToolOutput(
            tool_name=tool_config.name,
            command='',
            outputs=expected_outputs,
        )

    # 处理必需参数
    for param in tool_config.required_params:
        # 处理输入参数
        if param.param_type == ParamType.INPUT:
            if param.name in inputs:
                uid, key = inputs[param.name]
                file_path = get_tool_output(uid, key)
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
            output_cmd = param.command.format(output_dir=container_output_dir, name=sample_name)
            output_cmds.append(output_cmd)

    # 构建完整命令
    tool_command = tool_config.command_template.format(
        required_params=' '.join(input_cmds + output_cmds),
        optional_params=params or tool_config.optional_params,
    )

    logger.debug(mount_paths)

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

            logger.debug(f'执行命令: {tool_command}')
            logger.debug(f'挂载卷: {volume_binds}')
            # 执行Docker命令
            # run_container(
            #     tool_config.docker_config.image,
            #     commands=['/bin/bash', '-c', tool_command],
            #     volumes_binds=temp_volume_binds,
            #     **tool_config.docker_config.docker_settings,
            # )
            #
            # 复制结果到最终输出目录
            for output_name, host_output_path in expected_outputs.items():
                relative_path = host_output_path.relative_to(output_dir)
                temp_path = Path(temp_dir) / relative_path

                if temp_path.exists():
                    shutil.copy(temp_path, host_output_path)
                else:
                    logger.warning(f'临时输出文件不存在: {temp_path}')
    else:
        volume_binds = [
            f'{host_path}:{container_path}:rw' for container_path, host_path in mount_paths.items()
        ]
        logger.debug(f'挂载卷: {volume_binds}')
        logger.debug(f'执行命令: {tool_command}')

        # 直接使用最终输出目录
        # run_container(
        #     tool_config.docker_config.image,
        #     commands=['/bin/bash', '-c', tool_command],
        #     volumes_binds=volume_binds,
        #     **tool_config.docker_config.docker_settings,
        # )

    # 验证输出文件是否生成
    missing_outputs = [str(path) for name, path in expected_outputs.items() if not path.exists()]
    if missing_outputs:
        logger.warning(f'部分输出文件未生成: {missing_outputs}')

    # 返回结果
    return ToolOutput(tool_name=tool_config.name, command=tool_command, outputs=expected_outputs)


if __name__ == '__main__':
    fastp_config = DockerToolCreate(
        name='fastp',
        repository='staphb/fastp',
        tag='0.24.0',
        description='',
        command_template='fastp {required_params} {optional_params}&> /data/output/fastp.log',
        required_params=[
            ParamDefine(
                name='r1',
                command='-i {r1}',
                is_file=True,
                mount_path='/data/raw',
                param_type=ParamType.INPUT,
            ),
            ParamDefine(
                name='r2',
                command='-I {r2}',
                is_file=True,
                mount_path='/data/raw',
                param_type=ParamType.INPUT,
            ),
            ParamDefine(
                name='r1',
                command='-o {output_dir}/{name}_fastp_R1.fq.gz',
                is_file=True,
                param_type=ParamType.OUTPUT,
            ),
            ParamDefine(
                name='r2',
                command='-O {output_dir}/{name}_fastp_R2.fq.gz',
                is_file=True,
                param_type=ParamType.OUTPUT,
            ),
            ParamDefine(
                name='json_report',
                command='-p {output_dir}/{name}_fastp.json',
                is_file=True,
                param_type=ParamType.OUTPUT,
            ),
            ParamDefine(
                name='html_report',
                command='-h {output_dir}/{name}_fastp.html',
                is_file=True,
                param_type=ParamType.OUTPUT,
            ),
        ],
        optional_params='--thread 8',
        output_files=[
            OutputFile(
                name='r1', file_path='{output_dir}/{name}_fastp_R1.fq.gz', mount_path='/data/output'
            ),
            OutputFile(
                name='r2', file_path='{output_dir}/{name}_fastp_R2.fq.gz', mount_path='/data/output'
            ),
            OutputFile(
                name='json_report',
                file_path='{output_dir}/{name}_fastp.json',
                mount_path='/data/output',
            ),
            OutputFile(
                name='html_report',
                file_path='{output_dir}/{name}_fastp.html',
                mount_path='/data/output',
            ),
            OutputFile(name='log', file_path='{output_dir}/fastp.log', mount_path='/data/output'),
        ],
        mkdir_output=True,
        use_temp_dir=False,
        help_command='fastp --help',
    )

    tool_data = fastp_config.to_tool_data()

    docker_task(
        tool_config=tool_data,
        inputs={
            'r1': ('8c2d6f8b-69fa-4265-9f41-aca7807343a6', 'file_path'),
            'r2': ('72e050f3-9fc5-48d5-84d5-f474a5eb4a33', 'file_path'),
        },
        output_dir='/mnt/sda/gene/yeyu/scc/250421sc_96/01_clean/2503-A31',
        params='--thread 8',
        sample_name='2503-A31',
    )
