"""任务调度器.

负责工作流节点的排序、DAG 构建和任务分发
"""

import uuid
from typing import Optional

import rich
import rustworkx as rx
from celery import chain
from loguru import logger
from sqlmodel import Session
from uuid_extensions import uuid7str

from app.models.user import User
from app.schemas.react_flow import TaskInput, TaskNode, WorkflowDefinition
from app.utils.workflow import TaskContext, get_handler


def update_node_id(workflow: WorkflowDefinition) -> WorkflowDefinition:
    """刷新节点ID.

    用于从模板中实例化节点时刷新节点ID

    Args:
        workflow: 从模板中解析出的react-flow对象

    Returns:
        更新节点uid后的react-flow对象
    """
    for node in workflow.nodes:
        old_id = node.id
        new_id = uuid7str()
        # 不用刷新edge id, 这玩意没用
        for edge in workflow.edges:
            if edge.source == old_id:
                edge.source = new_id
                edge.sourceHandle = edge.sourceHandle.replace(old_id, new_id)

            elif edge.target == old_id:
                edge.target = new_id
                edge.targetHandle = edge.targetHandle.replace(old_id, new_id)

        node.id = new_id

    return workflow


def build_task_list(workflow: WorkflowDefinition) -> list[list[TaskNode]]:
    """对任务队列进行重组排序.

    分析react-flow原始数据, 转换为DAG
    剔除单独节点 (出入度为0) , 按深度分层排序

    Args:
        workflow: 原始react-flow数据

    Returns:
        排序后的任务节点队列
    """
    # rustworkx DAG对象
    graph = rx.PyDiGraph()
    # 节点uuid到rustworkx node_index映射
    node_index_map: dict[str, int] = {}

    # 构建TaskNode
    node_map: dict[str, TaskNode] = {}
    for node in workflow.nodes:
        node_map[node.id] = TaskNode(id=node.id, type=node.type, data=node.data)
    for edge in workflow.edges:
        target_param = edge.targetHandle.split('-in-')[-1]
        task_input = TaskInput(source_uid=edge.source, source_k=edge.sourceHandle.split('-out-')[-1])

        # 检查该参数是否已存在输入
        if target_param in node_map[edge.target].kwarg:
            existing = node_map[edge.target].kwarg[target_param]
            # 如果已经是列表, 追加
            if isinstance(existing, list):
                existing.append(task_input)
            # 如果是单个TaskInput, 转换为列表
            else:
                node_map[edge.target].kwarg[target_param] = [existing, task_input]
        else:
            # 首次添加, 保持单个TaskInput
            node_map[edge.target].kwarg[target_param] = task_input

    # DAG排序
    edge_set = [(edge.source, edge.target) for edge in workflow.edges]
    for source_uid, target_uid in edge_set:
        if source_uid not in node_index_map:
            node_index_map[source_uid] = graph.add_node(source_uid)

        if target_uid not in node_index_map:
            node_index_map[target_uid] = graph.add_node(target_uid)

        graph.add_edge(node_index_map[source_uid], node_index_map[target_uid], None)

    first_layers = [node for node in graph.node_indices() if graph.in_degree(node) == 0 and graph.out_degree(node) > 0]
    task_groups = rx.layers(graph, first_layers)
    return [[node_map[index] for index in sublist] for sublist in task_groups]


def send_task(
    session: Session,
    task_nodes: list[list[TaskNode]],
    sample_name: str,
    output_base: str,
    run_uid: uuid.UUID,
    owner: User,
    project_id: int,
    auto_fill: Optional[str] = None,
) -> None:
    """构建实际任务队列, 并发送至celery worker.

    Args:
        session: SQLModel 会话
        task_nodes: 排序后的任务节点队列
        sample_name: 此次分析的样本名称
        output_base: 根输出目录
        run_uid: 运行实例UID
        owner: 任务创建者
        project_id: 项目 ID
        auto_fill: 自动填充输入序列
    """
    rich.print(task_nodes)

    # 创建任务上下文
    context = TaskContext(
        session=session,
        sample_name=sample_name,
        output_base=output_base,
        run_uid=run_uid,
        owner=owner,
        project_id=project_id,
        auto_fill=auto_fill,
        task_index=1,
    )

    tasks = []
    for sub_group in task_nodes:
        sub_tasks = []
        for task_node in sub_group:
            # 获取对应的处理器
            handler = get_handler(task_node.type)

            if handler is None:
                logger.warning(f'未找到节点类型 {task_node.type} 的处理器')
                continue

            # 使用处理器处理任务节点
            celery_task, new_task_index = handler.handle(task_node, context)

            # 更新任务索引
            context.task_index = new_task_index

            # 添加到子任务列表 (如果任务不为空)
            if celery_task is not None:
                sub_tasks.append(celery_task)

        # 只有当子任务列表不为空时才添加到任务组
        if sub_tasks:
            # tasks.append(group(*sub_tasks))
            tasks.extend(sub_tasks)  # celery 目前不支持多个group

    # if context.downstream_files:
    #     handler = get_handler('summary')
    #     summary_task, _ = handler.handle(None, context)
    #     tasks.append(summary_task)

    # 构建并发送工作流
    if tasks:
        workflow = chain(*tasks)
        workflow.delay()
