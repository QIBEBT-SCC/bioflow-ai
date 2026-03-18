"""运行实例管理器.

负责批量创建和管理工作流运行实例
"""

import uuid
from typing import Optional

from fastapi import HTTPException
from loguru import logger
from sqlmodel import Session
from starlette.status import HTTP_404_NOT_FOUND

import app.crud.resource.sample as sample_crud
from app.crud import project as proj_crud
from app.crud import workflow as wf_crud
from app.models.user import User
from app.models.workflow import RunInstance, Status
from app.schemas.project_workflow import WorkflowRunResult
from app.utils.input_format import to_safe_path_segment
from app.utils.workflow.scheduler import build_task_list, send_task, update_node_id
from worker import monitor_redis_client


def create_runs_from_samples(
    session: Session,
    project_id: int,
    workflow_uid: uuid.UUID,
    sample_uids: list[uuid.UUID],
    current_user: User,
    run_name_prefix: Optional[str] = None,
) -> WorkflowRunResult:
    """从样本列表批量创建运行实例.

    为每个样本创建一个独立的 RunInstance, 并提交任务执行

    Args:
        session: 数据库会话
        project_id: 项目 ID
        workflow_uid: 工作流模板 UID
        sample_uids: 样本 UID 列表
        current_user: 当前用户
        run_name_prefix: 运行名称前缀 (可选)

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

        # 在 Redis 中初始化运行实例的任务统计信息 (使用 Hash 结构)
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
            f'{to_safe_path_segment(project.name)}/{to_safe_path_segment(workflow.name)}/{sample.sample_name}',
            run.uid,
            current_user,
            f'sample:{sample.uid}',
        )

        run_uids.append(run.uid)

    return WorkflowRunResult(run_uids=run_uids, count=len(run_uids))
