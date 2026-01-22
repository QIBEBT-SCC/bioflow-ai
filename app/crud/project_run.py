import uuid
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import Select
from sqlmodel import Session, col, select
from starlette.status import HTTP_404_NOT_FOUND

from app.models.project import Project
from app.models.resource import Sample
from app.models.user import User
from app.models.workflow import RunInstance, Status, Workflow
from app.schemas.project_workflow import WorkflowRunResult
from app.utils.workflow import build_task_list, send_task, update_node_id


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
    project_stmt: Select = select(Project).where(Project.id == project_id)
    project = session.exec(project_stmt).one_or_none()
    if not project:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Project not found')

    # 获取工作流模板
    workflow_stmt: Select = select(Workflow).where(Workflow.uid == workflow_uid)
    workflow = session.exec(workflow_stmt).one_or_none()
    if not workflow:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow not found')

    # 获取所有样本
    samples_stmt: Select = select(Sample).where(col(Sample.uid).in_(sample_uids))
    samples = session.exec(samples_stmt).all()

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

        # 构建任务列表并提交
        task_list = build_task_list(workflow_definition)

        # 提交任务
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

    session.commit()

    return WorkflowRunResult(run_uids=run_uids, count=len(run_uids))
