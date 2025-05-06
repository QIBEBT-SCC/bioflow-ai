from typing import Sequence, Optional

from fastapi import HTTPException
from sqlalchemy import Select
from sqlalchemy.orm import selectinload
from sqlmodel import Session, select, or_, col

from app.models.project import Project, ProjectUpdate


def insert(session: Session, project: Project) -> Project:
    session.add(project)
    session.commit()
    session.refresh(project)

    return project


def delete(session: Session, project: Project):
    raise NotImplementedError


def update(session: Session, project_id: int, project: ProjectUpdate) -> Project:
    db_project = get(session, project_id)
    if not db_project:
        raise HTTPException(status_code=404, detail='Project not find！')

    project_data = project.model_dump(exclude_unset=True)
    db_project.sqlmodel_update(project_data)
    session.add(db_project)
    session.commit()
    session.refresh(db_project)
    return db_project


def get_all(session: Session, limit: int, offset: int, user_id: int = 0) -> Sequence[Project]:
    statement: Select = (
        select(Project)
        .where(or_(Project.public, Project.owner_id == user_id))
        .options(selectinload(Project.tags), selectinload(Project.owner))  # type: ignore
        .offset(offset)
        .limit(limit)
    )
    return session.exec(statement).all()


def get_all_star(session: Session, limit: int, offset: int, user_id: int = 0) -> Sequence[Project]:
    statement: Select = (
        select(Project)
        .where(or_(Project.public, Project.owner_id == user_id))
        .where(Project.starred)
        .options(selectinload(Project.tags), selectinload(Project.owner))  # type: ignore
        .offset(offset)
        .limit(limit)
    )
    return session.exec(statement).all()


def get_all_my(session: Session, limit: int, offset: int, user_id: int) -> Sequence[Project]:
    statement: Select = (
        select(Project)
        .where(Project.owner_id == user_id)
        .options(selectinload(Project.tags), selectinload(Project.owner))  # type: ignore
        .offset(offset)
        .limit(limit)
    )
    return session.exec(statement).all()


def get(session: Session, project_id: int) -> Optional[Project]:
    statement: Select = select(Project).where(Project.id == project_id)
    return session.exec(statement).first()


def get_recent(session: Session, user_id: int) -> Optional[Project]:
    statement: Select = (
        select(Project).where(Project.owner_id == user_id).order_by(col(Project.update_time).desc())
    )
    return session.exec(statement).first()
