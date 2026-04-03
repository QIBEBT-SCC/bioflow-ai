"""运行实例管理器.

负责批量创建和管理工作流运行实例
"""

import uuid
from typing import Optional

from fastapi import HTTPException
from loguru import logger
from sqlmodel import Session
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND, HTTP_409_CONFLICT

import app.crud.resource.sample as sample_crud
from app.crud import project as proj_crud
from app.crud import run_instance as run_crud
from app.crud import workflow as wf_crud
from app.models.user import User
from app.models.workflow import ExecutionScope, RunInstance, Status
from app.schemas.project_workflow import WorkflowRunResult
from app.utils.input_format import to_safe_path_segment
from app.utils.workflow.scheduler import build_task_list, send_task, update_node_id
from app.utils.workflow.stats import reset_run_stats_in_redis
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
        # 检查该样本+工作流组合是否已有运行实例
        existing_run = run_crud.get_latest_run_by_sample_and_workflow(session, sample.uid, workflow_uid)

        if existing_run:
            # 如果正在运行, 拒绝重跑
            if existing_run.status == Status.RUNNING:
                raise HTTPException(
                    HTTP_409_CONFLICT,
                    f'Sample {sample.sample_name} already has a running instance, please wait for it to finish',
                )

            # 清除旧的执行数据并重用现有实例
            run_crud.clear_run_for_rerun(session, existing_run)
            logger.debug(f'重用运行实例 {existing_run.uid} (样本: {sample.sample_name})')
        else:
            # 更新工作流节点 ID
            workflow_definition = update_node_id(workflow.workflow)

            # 构建运行名称
            if run_name_prefix:
                run_name = f'{run_name_prefix}-{sample.sample_name}'
            else:
                run_name = f'{project.name}-{workflow.name}-{sample.sample_name}'

            # 创建新运行实例
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

        # 在 Redis 中初始化/重置运行实例的任务统计信息
        reset_run_stats_in_redis(monitor_redis_client, str(run.uid), total_tasks)

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
            project_id,
            f'sample:{sample.uid}',
        )

        run_uids.append(run.uid)

    return WorkflowRunResult(run_uids=run_uids, count=len(run_uids))


def create_project_level_run(
    session: Session,
    project_id: int,
    workflow_uid: uuid.UUID,
    current_user: User,
    run_name_prefix: Optional[str] = None,
) -> WorkflowRunResult:
    """创建项目级工作流运行实例.

    项目级工作流不绑定样本, 只运行一次, 使用 proj:keyword 引用

    Args:
        session: 数据库会话
        project_id: 项目 ID
        workflow_uid: 工作流模板 UID
        current_user: 当前用户
        run_name_prefix: 运行名称前缀 (可选)

    Returns:
        WorkflowRunResult: 包含创建的运行实例 UID

    Raises:
        HTTPException: 项目或工作流不存在, 或工作流不是项目级
    """
    # 获取项目
    project = proj_crud.get(session, current_user.id, project_id)
    if not project:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Project not found')

    # 获取工作流模板
    workflow = wf_crud.get(session, workflow_uid)
    if not workflow:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow not found')

    # 验证工作流是项目级
    if workflow.execution_scope != ExecutionScope.PROJECT_LEVEL:
        raise HTTPException(HTTP_400_BAD_REQUEST, 'Workflow is not project-level')

    # 更新工作流节点 ID
    workflow_definition = update_node_id(workflow.workflow)

    # 检查该项目+工作流组合是否已有运行实例
    existing_run = run_crud.get_latest_run_by_project_and_workflow(session, project_id, workflow_uid)

    if existing_run:
        # 如果正在运行, 拒绝重跑
        if existing_run.status == Status.RUNNING:
            raise HTTPException(
                HTTP_409_CONFLICT,
                'Project already has a running instance of this workflow',
            )

        # 清除旧的执行数据并重用现有实例
        run_crud.clear_run_for_rerun(session, existing_run)
        existing_run.workflow = workflow_definition
        session.add(existing_run)
        session.flush()
        run = existing_run
        logger.debug(f'重用项目级运行实例 {run.uid}')
    else:
        # 构建运行名称
        if run_name_prefix:
            run_name = f'{run_name_prefix}-{project.name}'
        else:
            run_name = f'{project.name}-{workflow.name}'

        # 创建新运行实例 (sample_uid=None)
        run = RunInstance(
            project_id=project_id,
            workflow_uid=workflow_uid,
            sample_uid=None,  # 项目级工作流不绑定样本
            owner_id=current_user.id,
            workflow=workflow_definition,
            name=run_name,
            status=Status.WAITING,
        )

        session.add(run)
        session.flush()

    # 构建任务列表
    task_list = build_task_list(workflow_definition)

    # 计算总任务数
    total_tasks = sum(len(sub_group) for sub_group in task_list)

    # 在 Redis 中初始化/重置运行实例的任务统计信息
    reset_run_stats_in_redis(monitor_redis_client, str(run.uid), total_tasks)

    # 提交当前运行实例到数据库
    session.commit()
    logger.debug(f'项目级运行实例 {run.uid} 已提交到数据库')

    # 提交任务到 Celery (无样本上下文)
    send_task(
        session,
        task_list,
        project.name,  # 使用项目名作为 sample_name
        f'{to_safe_path_segment(project.name)}/{to_safe_path_segment(workflow.name)}',
        run.uid,
        current_user,
        project_id,
        None,  # auto_fill=None 表示项目级执行
    )

    return WorkflowRunResult(run_uids=[run.uid], count=1)
