import uuid

from fastapi import HTTPException
from sqlalchemy import delete
from sqlmodel import Session, col, select
from starlette.status import HTTP_404_NOT_FOUND, HTTP_409_CONFLICT

from app.models.project import ProjectWorkflow
from app.models.workflow import RunInstance, TaskInstance, TaskMonitor, Workflow
from app.schemas.project_workflow import ProjectWorkflowPublic


def add_workflow_to_project(session: Session, project_id: int, workflow_uid: uuid.UUID) -> ProjectWorkflow:
    """将工作流模板添加到项目中.

    Args:
        session: 数据库会话
        project_id: 项目 ID
        workflow_uid: 工作流模板 UID

    Returns:
        ProjectWorkflow: 创建的关联记录

    Raises:
        HTTPException: 工作流不存在或已存在于项目中
    """
    # 检查工作流是否存在
    workflow_stmt = select(Workflow).where(Workflow.uid == workflow_uid)
    workflow = session.exec(workflow_stmt).one_or_none()
    if not workflow:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow not found')

    # 检查是否已存在
    existing_stmt = select(ProjectWorkflow).where(
        ProjectWorkflow.project_id == project_id, ProjectWorkflow.workflow_uid == workflow_uid
    )
    existing = session.exec(existing_stmt).one_or_none()
    if existing:
        raise HTTPException(HTTP_409_CONFLICT, 'Workflow already exists in this project')

    # 创建关联
    project_workflow = ProjectWorkflow(project_id=project_id, workflow_uid=workflow_uid)
    session.add(project_workflow)
    session.commit()
    session.refresh(project_workflow)

    return project_workflow


def remove_workflow_from_project(session: Session, project_id: int, workflow_uid: uuid.UUID) -> None:
    """从项目中移除工作流模板.

    Args:
        session: 数据库会话
        project_id: 项目 ID
        workflow_uid: 工作流模板 UID

    Raises:
        HTTPException: 工作流不在项目中
    """
    # 手动处理运行实例及其关联数据的级联删除
    run_instance_uids = session.exec(
        select(RunInstance.uid).where(
            col(RunInstance.project_id) == project_id, col(RunInstance.workflow_uid) == workflow_uid
        )
    ).all()

    if run_instance_uids:
        # 获取所有关联的任务实例 UID
        task_instance_uids = session.exec(
            select(TaskInstance.uid).where(col(TaskInstance.instance_uid).in_(run_instance_uids))
        ).all()

        if task_instance_uids:
            # 1. 删除任务监控日志 (TaskMonitor)
            session.exec(delete(TaskMonitor).where(col(TaskMonitor.task_uid).in_(task_instance_uids)))

        # 2. 删除任务实例 (TaskInstance)
        session.exec(delete(TaskInstance).where(col(TaskInstance.instance_uid).in_(run_instance_uids)))

        # 3. 删除运行实例 (RunInstance)
        session.exec(delete(RunInstance).where(col(RunInstance.uid).in_(run_instance_uids)))

    stmt = select(ProjectWorkflow).where(
        ProjectWorkflow.project_id == project_id, ProjectWorkflow.workflow_uid == workflow_uid
    )
    project_workflow = session.exec(stmt).one_or_none()

    if not project_workflow:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow not found in this project')

    session.delete(project_workflow)

    session.commit()


def get_project_workflows(
    session: Session, project_id: int, limit: int = 20, offset: int = 0
) -> list[ProjectWorkflowPublic]:
    """获取项目中的所有工作流模板.

    Args:
        session: 数据库会话
        project_id: 项目 ID
        limit: 限制数量
        offset: 偏移量

    Returns:
        list[ProjectWorkflowPublic]: 工作流列表
    """
    # 联合查询获取工作流详细信息
    stmt = (
        select(ProjectWorkflow, Workflow)
        .join(Workflow, col(ProjectWorkflow.workflow_uid) == Workflow.uid)
        .where(ProjectWorkflow.project_id == project_id)
        .offset(offset)
        .limit(limit)
    )

    results = session.exec(stmt).all()

    return [
        ProjectWorkflowPublic(
            # pyrefly: ignore [bad-argument-type]
            workflow_uid=pw.workflow_uid,
            workflow_name=wf.name,
            execution_scope=wf.execution_scope,
            import_time=pw.import_time,
            enabled=pw.enabled,
            auto_summary=wf.auto_summary,
        )
        for pw, wf in results
    ]


def is_workflow_in_project(session: Session, project_id: int, workflow_uid: uuid.UUID) -> bool:
    """检查工作流是否在项目中.

    Args:
        session: 数据库会话
        project_id: 项目 ID
        workflow_uid: 工作流模板 UID

    Returns:
        bool: 是否存在
    """
    stmt = select(ProjectWorkflow).where(
        ProjectWorkflow.project_id == project_id, ProjectWorkflow.workflow_uid == workflow_uid
    )
    return session.exec(stmt).one_or_none() is not None


def toggle_workflow_enabled(
    session: Session, project_id: int, workflow_uid: uuid.UUID, *, enabled: bool
) -> ProjectWorkflow:
    """启用或禁用项目中的工作流.

    Args:
        session: 数据库会话
        project_id: 项目 ID
        workflow_uid: 工作流模板 UID
        enabled: 是否启用

    Returns:
        ProjectWorkflow: 更新后的关联记录

    Raises:
        HTTPException: 工作流不在项目中
    """
    stmt = select(ProjectWorkflow).where(
        ProjectWorkflow.project_id == project_id, ProjectWorkflow.workflow_uid == workflow_uid
    )
    project_workflow = session.exec(stmt).one_or_none()

    if not project_workflow:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow not found in this project')

    project_workflow.enabled = enabled
    session.add(project_workflow)
    session.commit()
    session.refresh(project_workflow)

    return project_workflow
