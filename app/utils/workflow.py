import uuid
from typing import Optional

import rich
import rustworkx as rx
from celery import chain, group
from fastapi import HTTPException
from loguru import logger
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from sqlmodel import Session
from starlette.status import HTTP_404_NOT_FOUND
from uuid_extensions import uuid7str

import app.crud.resource.bio_db as db_crud
import app.crud.resource.sample as sample_crud
import app.crud.task_instance as task_crud
import app.crud.tools.tool as tool_crud
from app.core.config import get_settings
from app.crud import project as proj_crud
from app.crud import workflow as wf_crud
from app.models.resource import SampleFileType
from app.models.user import User
from app.models.workflow import RunInstance, Status, TaskInstance
from app.schemas.common import Statistics
from app.schemas.project_workflow import WorkflowRunResult
from app.schemas.react_flow import TaskInput, TaskNode, WorkflowDefinition
from app.tasks.load_tasks.load_file import load_file_task, load_sequence_task
from app.tasks.load_tasks.load_genome import load_genome_task
from app.tasks.load_tasks.static_value import set_value_task
from app.tasks.run_tool import docker_task
from worker import monitor_redis_client


def update_node_id(workflow: WorkflowDefinition) -> WorkflowDefinition:
    """刷新节点ID

    用于从模板中实例化节点时刷新节点ID

    Args:
        workflow: 从模板中解析出的react-flow对象

    Returns:
        更新节点uid后的react-flow对象
    """
    for node in workflow.nodes:
        old_id = node.id
        new_id = uuid7str()
        # 不用刷新edge id，这玩意没用
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
    """对任务队列进行重组排序

    分析react-flow原始数据，转换为DAG
    剔除单独节点（出入度为0），按深度分层排序

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
        node_map[node.id] = TaskNode(
            id=node.id, type=node.type, resource_uid=node.data.resource_uid, args=node.data.args
        )
    for edge in workflow.edges:
        node_map[edge.target].kwarg[edge.targetHandle.split('-in-')[-1]] = TaskInput(
            source_uid=edge.source, source_k=edge.sourceHandle.split('-out-')[-1]
        )

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
    auto_fill: Optional[str] = None,
) -> None:
    """构建实际任务队列，并发送至celery worker

    Args:
        session: SQLModel 会话
        task_nodes: 排序后的任务节点队列
        sample_name: 此次分析的样本名称
        output_base: 根输出目录
        run_uid: 运行实例UID
        owner: 任务创建者
        auto_fill: 自动填充输入序列
    """

    rich.print(task_nodes)

    tasks = []
    task_index = 1
    for sub_group in task_nodes:
        sub_tasks = []
        for task_node in sub_group:
            match task_node.type:
                case 'tool':
                    assert task_node.resource_uid is not None
                    tool_data = tool_crud.get(session, task_node.resource_uid)

                    task_instance = TaskInstance(
                        uid=uuid.UUID(task_node.id),
                        instance_uid=run_uid,
                        tool_uid=uuid.UUID(task_node.resource_uid),
                        owner_id=owner.id,
                        name=tool_data.name,
                        status=Status.WAITING,
                    )
                    task_crud.insert(session, task_instance)

                    sub_tasks.append(
                        docker_task.s(
                            tool_uid=task_node.resource_uid,
                            inputs=task_node.kwarg,
                            output_dir=f'{get_settings().persistent.STORAGE_PATH}/{owner.username}/{output_base}/{task_index}_{tool_data.name}',
                            params=task_node.args,
                            sample_name=sample_name,
                        ).set(task_id=task_node.id)
                    )
                    task_index += 1
                case 'resource_db':
                    task_instance = TaskInstance(
                        uid=uuid.UUID(task_node.id),
                        instance_uid=run_uid,
                        owner_id=owner.id,
                        name='read_file',
                        status=Status.WAITING,
                    )
                    task_crud.insert(session, task_instance)

                    db_path = db_crud.get(session, int(task_node.resource_uid)).path
                    sub_tasks.append(load_file_task.s(file_path=db_path).set(task_id=task_node.id))
                case 'resource_sequence':
                    task_instance = TaskInstance(
                        uid=uuid.UUID(task_node.id),
                        instance_uid=run_uid,
                        owner_id=owner.id,
                        name='read_file',
                        status=Status.WAITING,
                    )
                    task_crud.insert(session, task_instance)

                    if auto_fill:
                        sample_uid = auto_fill.split(':')[1]
                        sample = sample_crud.get_with_files(session, sample_uid)  # type: ignore

                        r1_path = next(
                            (f.file_path for f in sample.sample_files if f.data_type == SampleFileType.SEQUENCING_R1),
                            None,
                        )
                        r2_path = next(
                            (f.file_path for f in sample.sample_files if f.data_type == SampleFileType.SEQUENCING_R2),
                            None,
                        )

                        if r1_path is None:
                            raise Exception('Sample missing r1 sequencing file')
                        if r2_path is None:
                            raise Exception('Sample missing r2 sequencing file')

                        sub_tasks.append(
                            load_sequence_task.s(args={'r1': r1_path, 'r2': r2_path}).set(task_id=task_node.id)
                        )
                    else:
                        sub_tasks.append(load_sequence_task.s(args=task_node.args).set(task_id=task_node.id))
                case 'resource_genome':
                    task_instance = TaskInstance(
                        uid=uuid.UUID(task_node.id),
                        instance_uid=run_uid,
                        owner_id=owner.id,
                        name='load_reference_genome',
                        status=Status.WAITING,
                    )
                    task_crud.insert(session, task_instance)

                    sub_tasks.append(
                        load_genome_task.s(
                            inputs=task_node.kwarg,
                            required_index=task_node.args['requiredIndex'],
                        ).set(task_id=task_node.id)
                    )
                case 'resource_file':
                    task_instance = TaskInstance(
                        uid=uuid.UUID(task_node.id),
                        instance_uid=run_uid,
                        owner_id=owner.id,
                        name='read_file',
                        status=Status.WAITING,
                    )
                    task_crud.insert(session, task_instance)

                    sub_tasks.append(load_file_task.s(file_path=task_node.args).set(task_id=task_node.id))
                case 'value_string':
                    task_instance = TaskInstance(
                        uid=uuid.UUID(task_node.id),
                        instance_uid=run_uid,
                        owner_id=owner.id,
                        name='value',
                        status=Status.WAITING,
                    )
                    task_crud.insert(session, task_instance)

                    sub_tasks.append(
                        set_value_task.s(value=task_node.args, value_type='string').set(task_id=task_node.id)
                    )
                case 'process_copy2folder':
                    pass
                case _:
                    logger.warning(f'unknown node type {task_node.type}')
                    pass

        tasks.append(group(*sub_tasks))

    workflow = chain(*tasks)
    workflow.delay()


def create_runs_from_samples(
    session: Session,
    project_id: int,
    workflow_uid: uuid.UUID,
    sample_uids: list[uuid.UUID],
    current_user: User,
    run_name_prefix: Optional[str] = None,
) -> WorkflowRunResult:
    """从样本列表批量创建运行实例

    为每个样本创建一个独立的 RunInstance，并提交任务执行

    Args:
        session: 数据库会话
        project_id: 项目 ID
        workflow_uid: 工作流模板 UID
        sample_uids: 样本 UID 列表
        current_user: 当前用户
        run_name_prefix: 运行名称前缀（可选）

    Returns:
        WorkflowRunResult: 包含创建的运行实例 UID 列表

    Raises:
        HTTPException: 项目、工作流或样本不存在
    """
    # 获取项目
    project = proj_crud.get(session, current_user.id, project_id)
    if not project:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Project not found')

    # 获取工作流模板
    workflow = wf_crud.get(session, workflow_uid)
    if not workflow:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow not found')

    # 获取所有样本
    samples = sample_crud.get_all(session, sample_uids)
    if len(samples) != len(sample_uids):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Some samples not found')

    # 为每个样本创建运行实例
    run_uids = []

    for sample in samples:
        # 更新工作流节点 ID
        workflow_definition = update_node_id(workflow.workflow)

        # 构建运行名称
        if run_name_prefix:
            run_name = f'{run_name_prefix}-{sample.sample_name}'
        else:
            run_name = f'{project.name}-{workflow.name}-{sample.sample_name}'

        # 创建运行实例
        run = RunInstance(
            project_id=project_id,
            workflow_uid=workflow_uid,
            sample_uid=sample.uid,
            owner_id=current_user.id,
            workflow=workflow_definition,
            name=run_name,
            status=Status.WAITING,
        )

        session.add(run)
        session.flush()  # 获取 run.uid

        # 构建任务列表
        task_list = build_task_list(workflow_definition)

        # 计算总任务数
        total_tasks = sum(len(sub_group) for sub_group in task_list)

        # 在 Redis 中初始化运行实例的任务统计信息（使用 Hash 结构）
        try:
            redis_key = f'run-stats-{run.uid}'
            monitor_redis_client.hset(
                redis_key,
                mapping={
                    'total': total_tasks,
                    'waiting': total_tasks,
                    'running': 0,
                    'success': 0,
                    'error': 0,
                },
            )
            # 设置过期时间为 24 小时
            monitor_redis_client.expire(redis_key, 86400)
            logger.debug(f'已初始化运行实例 {run.uid} 的统计信息: total={total_tasks}')
        except Exception as e:
            logger.error(f'初始化 Redis 统计信息失败: {e}')

        # 提交当前运行实例到数据库
        session.commit()
        logger.debug(f'运行实例 {run.uid} 已提交到数据库')

        # 提交任务到 Celery
        send_task(
            session,
            task_list,
            sample.sample_name,
            f'{project.name}/{workflow.name}/{sample.sample_name}',
            run.uid,
            current_user,
            f'sample:{sample.uid}',
        )

        run_uids.append(run.uid)

    return WorkflowRunResult(run_uids=run_uids, count=len(run_uids))


async def get_run_stats_from_redis(redis_client: AsyncRedis, run_uid: str) -> Optional[Statistics]:
    """从 Redis Hash 中获取运行实例的统计信息（异步版本）

    Args:
        redis_client: 异步 Redis 客户端
        run_uid: 运行实例 UID

    Returns:
        Statistics: 统计信息对象，如果不存在则返回 None
    """
    redis_key = f'run-stats-{run_uid}'

    try:
        if not await redis_client.exists(redis_key):
            return None

        stats = await redis_client.hgetall(redis_key)
        if not stats:
            return None

        return Statistics(
            total=int(stats.get(b'total', 0)),
            waiting=int(stats.get(b'waiting', 0)),
            running=int(stats.get(b'running', 0)),
            success=int(stats.get(b'success', 0)),
            error=int(stats.get(b'error', 0)),
        )
    except Exception as e:
        logger.error(f'从 Redis 读取统计信息失败: {e}')
        return None


def get_run_stats_from_redis_sync(redis_client: Redis, run_uid: str) -> Optional[Statistics]:
    """从 Redis Hash 中获取运行实例的统计信息（同步版本）

    Args:
        redis_client: 同步 Redis 客户端
        run_uid: 运行实例 UID

    Returns:
        Statistics: 统计信息对象，如果不存在则返回 None
    """
    redis_key = f'run-stats-{run_uid}'

    try:
        if not redis_client.exists(redis_key):
            return None

        stats = redis_client.hgetall(redis_key)
        if not stats:
            return None

        return Statistics(
            total=int(stats.get(b'total', 0)),
            waiting=int(stats.get(b'waiting', 0)),
            running=int(stats.get(b'running', 0)),
            success=int(stats.get(b'success', 0)),
            error=int(stats.get(b'error', 0)),
        )
    except Exception as e:
        logger.error(f'从 Redis 读取统计信息失败: {e}')
        return None
