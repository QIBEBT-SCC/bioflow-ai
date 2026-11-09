import json
from typing import Any

from loguru import logger

from bioflow_ai.infrastructure.redis.client import sync_monitor_redis_client
from bioflow_ai.schemas.react_flow import TaskInput
from bioflow_ai.schemas.tool_config import FileMount, ImageConfig, ToolType
from bioflow_ai.schemas.workflow_config import ToolOutput
from bioflow_ai.schemas.workflow_run import NodeRunUpdate
from bioflow_ai.services.tools.execution import (
    DockerToolExecutionConfig,
    ResolvedToolInput,
    execute_docker_tool,
    prepare_docker_tool_execution,
)
from bioflow_ai.services.workflow.execution_context import (
    current_node_uid,
    current_run_provenance,
    get_node_output,
    get_tool_data,
    tool_input_provenance,
    update_node,
)


def _resolve_task_inputs(
    file_mounts: list[FileMount],
    inputs: dict[str, TaskInput],
    tool_type: ToolType = ToolType.DOCKER,
) -> dict[str, ResolvedToolInput]:
    """Resolve workflow task references before entering the persistence-free executor."""
    resolved_inputs: dict[str, ResolvedToolInput] = {}
    for file in file_mounts:
        if file.file_type != 'INPUT':
            continue
        task_input = inputs[file.name]
        value, source_sample_name = get_node_output(task_input.source_uid, task_input.source_k)
        resolved_inputs[file.name] = ResolvedToolInput(
            value=value,
            sample_name=source_sample_name,
            **(
                tool_input_provenance(task_input.source_uid, writable=file.is_passthrough)
                if tool_type == ToolType.DOCKER_PASSTHROUGH
                else {}
            ),
        )
    return resolved_inputs


def execute_tool(
    *,
    tool_uid: str,
    inputs: dict[str, TaskInput],
    output_dir: str,
    sample_name: str,
    params: str = '',
) -> ToolOutput:
    """通用Docker工具执行任务.

    Args:
        tool_uid: 工具uid, 用于从数据库获取具体信息
        inputs: 输入参数字典, 格式为 {参数名: 上游节点输出引用}
        output_dir: 输出目录
        params: 额外参数
        sample_name: 样本名称

    Returns:
        ToolOutput: 工具输出结果
    """
    node_uid = current_node_uid()
    tool_config = get_tool_data(tool_uid)
    logger.debug(f'node {node_uid} start: {tool_config.name}')
    resolved_inputs = _resolve_task_inputs(tool_config.file_mounts, inputs, tool_config.tool_type)
    config = DockerToolExecutionConfig(
        image=ImageConfig.model_validate(tool_config.image.image),
        tool_type=tool_config.tool_type,
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
        **(current_run_provenance() if tool_config.tool_type == ToolType.DOCKER_PASSTHROUGH else {}),
    )

    data = NodeRunUpdate(
        base_dir=execution.output_dir,
        commands=f'docker run --rm '
        f'{" ".join([f"-v {bind.source}:{bind.target}:rw" for bind in execution.volume_binds])} '
        f'{execution.image.get_image()} /bin/bash -c "{execution.command}"',
    )
    update_node(data)

    redis_key = f'node-monitor-{node_uid}'

    def save_monitor(stats: dict[str, Any] | None) -> None:
        sync_monitor_redis_client.rpush(redis_key, json.dumps(stats))

    result = execute_docker_tool(execution, monitor=save_monitor)
    sync_monitor_redis_client.expire(redis_key, 60 * 60 * 24)
    return result
