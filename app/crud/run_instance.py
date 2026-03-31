import uuid
from collections.abc import Sequence
from typing import Optional

from sqlalchemy import Select, delete
from sqlalchemy.orm import selectinload
from sqlmodel import Session, case, col, func, select

from app.models.workflow import RunInstance, Status, TaskInstance
from app.schemas.common import Statistics
from app.schemas.run_instance import RunUpdate


def insert(session: Session, run: RunInstance) -> RunInstance:
    session.add(run)
    session.commit()
    session.refresh(run)

    return run


def update(session: Session, uid: str, run_update: RunUpdate) -> RunInstance:
    db_run = get(session, uid)
    update_data = run_update.model_dump(exclude_unset=True)
    db_run.sqlmodel_update(update_data)
    session.add(db_run)
    session.commit()
    session.refresh(db_run)
    return db_run


def get_run_counts(session: Session, user_id: int):
    statement: Select = select(func.count()).select_from(RunInstance).where(col(RunInstance.owner_id) == user_id)

    return session.exec(statement).one()


def get_run_status(session: Session, user_id: int) -> Statistics:
    """Get statistics of run instances by their status."""
    stats: Select = (
        select(  # type: ignore
            func.count().label('total'),
            func.sum(case((RunInstance.status == Status.SUCCESS, 1), else_=0)).label('success'),
            func.sum(case((RunInstance.status == Status.ERROR, 1), else_=0)).label('error'),
            func.sum(case((RunInstance.status == Status.RUNNING, 1), else_=0)).label('running'),
            func.sum(case((RunInstance.status == Status.WAITING, 1), else_=0)).label('waiting'),
        )
        .select_from(RunInstance)
        .where(col(RunInstance.owner_id) == user_id)
    )

    total, success, error, running, waiting = session.exec(stats).one()
    return Statistics(total=total, success=success, error=error, running=running, waiting=waiting)


def get_all_runs(session: Session, limit: int, offset: int, user_id: int) -> Sequence[RunInstance]:
    statement: Select = (
        select(RunInstance)
        .where(RunInstance.owner_id == user_id)
        .options(selectinload(RunInstance.owner), selectinload(RunInstance.tasks))  # type: ignore
        .order_by(col(RunInstance.create_time).desc())
        .limit(limit)
        .offset(offset)
    )

    return session.exec(statement).all()


def get(session: Session, uid: str) -> RunInstance:
    statement: Select = (
        select(RunInstance)
        .where(RunInstance.uid == uid)
        .options(selectinload(RunInstance.owner), selectinload(RunInstance.tasks))  # type: ignore
    )
    return session.exec(statement).one()


def get_run_task_stats(session: Session, uid: str) -> Statistics:
    """获取运行实例的任务统计信息."""
    statement: Select = (
        select(  # type: ignore
            func.count().label('total'),
            func.sum(case((col(TaskInstance.status) == Status.SUCCESS, 1), else_=0)).label('success'),
            func.sum(case((col(TaskInstance.status) == Status.ERROR, 1), else_=0)).label('error'),
            func.sum(case((col(TaskInstance.status) == Status.RUNNING, 1), else_=0)).label('running'),
            func.sum(case((col(TaskInstance.status) == Status.WAITING, 1), else_=0)).label('waiting'),
        )
        .select_from(TaskInstance)
        .where(col(TaskInstance.instance_uid) == uid)
    )

    total, success, error, running, waiting = session.exec(statement).one()
    return Statistics(total=total, success=success, error=error, running=running, waiting=waiting)


def get_runs_by_project(session: Session, project_id: int, limit: int = 20, offset: int = 0) -> Sequence[RunInstance]:
    """获取项目的所有运行实例."""
    statement: Select = (
        select(RunInstance)
        .where(RunInstance.project_id == project_id)
        .options(selectinload(RunInstance.owner), selectinload(RunInstance.tasks))  # type: ignore
        .order_by(col(RunInstance.create_time).desc())
        .limit(limit)
        .offset(offset)
    )

    return session.exec(statement).all()


def get_runs_by_sample(session: Session, sample_uid: str, limit: int = 20, offset: int = 0) -> Sequence[RunInstance]:
    """获取样本的所有运行实例.

    Args:
        session: 数据库会话
        sample_uid: 样本 UID
        limit: 限制数量
        offset: 偏移量

    Returns:
        运行实例列表
    """
    statement: Select = (
        select(RunInstance)
        .where(RunInstance.sample_uid == sample_uid)
        .options(selectinload(RunInstance.owner), selectinload(RunInstance.tasks))  # type: ignore
        .order_by(col(RunInstance.create_time).desc())
        .limit(limit)
        .offset(offset)
    )

    return session.exec(statement).all()


def get_project_run_count(session: Session, project_id: int) -> int:
    """获取项目的运行实例数量.

    Args:
        session: 数据库会话
        project_id: 项目 ID

    Returns:
        运行实例数量
    """
    statement: Select = select(func.count()).select_from(RunInstance).where(RunInstance.project_id == project_id)

    return session.exec(statement).one()


def get_latest_run_by_sample_and_workflow(
    session: Session, sample_uid: uuid.UUID, workflow_uid: uuid.UUID
) -> Optional[RunInstance]:
    """获取样本和工作流组合的最新运行实例.

    Args:
        session: 数据库会话
        sample_uid: 样本 UID
        workflow_uid: 工作流模板 UID

    Returns:
        最新的运行实例, 若不存在则返回 None
    """
    statement: Select = (
        select(RunInstance)
        .where(RunInstance.sample_uid == sample_uid)
        .where(RunInstance.workflow_uid == workflow_uid)
        .options(selectinload(RunInstance.tasks))  # type: ignore
        .order_by(col(RunInstance.create_time).desc())
        .limit(1)
    )
    return session.exec(statement).first()


def clear_run_for_rerun(session: Session, run: RunInstance) -> None:
    """清除运行实例的执行数据.

    - 删除所有 TaskInstance 记录 (级联删除 TaskMonitor)
    - 重置运行状态为 WAITING
    - 清除 start_time 和 end_time

    Args:
        session: 数据库会话
        run: 要清除的运行实例
    """
    session.exec(delete(TaskInstance).where(col(TaskInstance.instance_uid) == run.uid))  # type: ignore
    run.status = Status.WAITING
    run.start_time = None
    run.end_time = None
    session.add(run)
    session.flush()
