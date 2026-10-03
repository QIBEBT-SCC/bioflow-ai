import json
from typing import Any

from loguru import logger

from bioflow_ai.runtime.celery_app import monitor_redis_client
from bioflow_ai.schemas.react_flow import TaskInput
from bioflow_ai.schemas.tool_config import FileMount, ImageConfig
from bioflow_ai.schemas.workflow_config import ToolOutput
from bioflow_ai.schemas.workflow_run import NodeRunUpdate
from bioflow_ai.services.tools.execution import (
    DockerToolExecutionConfig,
    ResolvedToolInput,
    execute_docker_tool,
    prepare_docker_tool_execution,
)
from bioflow_ai.services.workflow.execution_context import get_node_output, get_tool_data, update_node


def _resolve_task_inputs(
    file_mounts: list[FileMount],
    inputs: dict[str, TaskInput],
) -> dict[str, ResolvedToolInput]:
    """Resolve workflow task references before entering the persistence-free executor."""
    resolved_inputs: dict[str, ResolvedToolInput] = {}
    for file in file_mounts:
        if file.file_type != 'INPUT':
            continue
        task_input = inputs[file.name]
        value, source_sample_name = get_node_output(task_input.source_uid, task_input.source_k)
        resolved_inputs[file.name] = ResolvedToolInput(value=value, sample_name=source_sample_name)
    return resolved_inputs


def execute_tool(
    *,
    node_run_id: str,
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
    tool_config = get_tool_data(tool_uid)
    logger.debug(f'node {node_run_id} start: {tool_config.name}')
    resolved_inputs = _resolve_task_inputs(tool_config.file_mounts, inputs)
    config = DockerToolExecutionConfig(
        image=ImageConfig.model_validate(tool_config.image.image),
        command_template=tool_config.command_template,
        dynamic_params=tool_config.dynamic_params,
        immutable_static_params=tool_config.immutable_static_params,
        modifiable_static_params=tool_config.modifiable_static_params,
        file_mounts=tool_config.file_mounts,
    )
    execution = prepare_docker_tool_execution(
        config,
        resolved_inputs,
        output_dir,
        sample_name,
        params,
    )

    data = NodeRunUpdate(
        commands=f'docker run --rm '
        f'{" ".join([f"-v {bind.source}:{bind.target}:rw" for bind in execution.volume_binds])} '
        f'{execution.image.get_image()} /bin/bash -c "{execution.command}"'
    )
    update_node(node_run_id, data)

    redis_key = f'node-monitor-{node_run_id}'

    def save_monitor(stats: dict[str, Any] | None) -> None:
        monitor_redis_client.rpush(redis_key, json.dumps(stats))

    result = execute_docker_tool(execution, monitor=save_monitor)
    monitor_redis_client.expire(redis_key, 60 * 60 * 24)
    return result
