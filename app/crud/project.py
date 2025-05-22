from typing import Optional

from fastapi import HTTPException
from sqlalchemy import Select
from sqlalchemy.orm import selectinload
from sqlmodel import Session, select, or_, col

from app.models.project import Project, ProjectUpdate, ProjectStar, ProjectPublic


def insert(session: Session, project: Project) -> Project:
    session.add(project)
    session.commit()
    session.refresh(project)

    return project


def delete(session: Session, project: Project):
    raise NotImplementedError


def update(session: Session, project_id: int, project: ProjectUpdate) -> Project:
    raise NotImplementedError


def get_star_ids(session: Session, user_id: int, project_ids: list[int]) -> set[int]:
    star_stmt: Select = select(ProjectStar.project_id).where(
        ProjectStar.user_id == user_id, col(ProjectStar.project_id).in_(project_ids)
    )
    starred_ids = set(r for r in session.exec(star_stmt).all())

    return starred_ids


def star_project(session: Session, user_id: int, project_id: int):
    statement: Select = select(ProjectStar).where(
        ProjectStar.user_id == user_id, ProjectStar.project_id == project_id
    )
    exists = session.exec(statement).first()
    if not exists:
        star = ProjectStar(user_id=user_id, project_id=project_id)
        session.add(star)
        session.commit()


def unstar_project(session: Session, user_id: int, project_id: int):
    statement: Select = select(ProjectStar).where(
        ProjectStar.user_id == user_id, ProjectStar.project_id == project_id
    )
    star = session.exec(statement).first()
    if star:
        session.delete(star)
        session.commit()


def get_all(session: Session, limit: int, offset: int, user_id: int) -> list[ProjectPublic]:
    statement: Select = (
        select(Project)
        .where(or_(Project.public, Project.owner_id == user_id))
        .options(selectinload(Project.tags), selectinload(Project.owner))  # type: ignore
        .offset(offset)
        .limit(limit)
    )
    projects = session.exec(statement).all()

    project_ids = [p.id for p in projects]
    starred_ids = get_star_ids(session, user_id, project_ids)

    result = []
    for project in projects:
        project_public = ProjectPublic.from_orm(
            project, update={'starred': project.id in starred_ids}
        )
        result.append(project_public)
    return result


def get_all_star(session: Session, limit: int, offset: int, user_id: int) -> list[ProjectPublic]:
    starred_ids = (
        select(ProjectStar.project_id)
        .where(ProjectStar.user_id == user_id)
        .offset(offset)
        .limit(limit)
    )

    statement: Select = (
        select(Project)
        .where(col(Project.id).in_(starred_ids))
        .options(selectinload(Project.tags), selectinload(Project.owner))  # type: ignore
    )

    projects = session.exec(statement).all()
    return [ProjectPublic.from_orm(p, update={'starred': True}) for p in projects]


def get_all_my(session: Session, limit: int, offset: int, user_id: int) -> list[ProjectPublic]:
    statement: Select = (
        select(Project)
        .where(Project.owner_id == user_id)
        .options(selectinload(Project.tags), selectinload(Project.owner))  # type: ignore
        .offset(offset)
        .limit(limit)
    )
    projects = session.exec(statement).all()

    project_ids = [p.id for p in projects]
    starred_ids = get_star_ids(session, user_id, project_ids)

    result = []
    for project in projects:
        project_public = ProjectPublic.from_orm(
            project, update={'starred': project.id in starred_ids}
        )
        result.append(project_public)
    return result


def get(session: Session, user_id: int, project_id: int) -> ProjectPublic:
    statement: Select = select(Project).where(Project.id == project_id)
    project: Project = session.exec(statement).one()

    statement: Select = select(ProjectStar).where(
        ProjectStar.user_id == user_id, ProjectStar.project_id == project.id
    )
    starred = session.exec(statement).first() is not None
    return ProjectPublic.from_orm(project, update={'starred': starred})


def get_recent(session: Session, user_id: int) -> Optional[ProjectPublic]:
    statement: Select = (
        select(Project).where(Project.owner_id == user_id).order_by(col(Project.update_time).desc())
    )
    project: Project = session.exec(statement).first()

    if project is None:
        return None

    statement: Select = select(ProjectStar).where(
        ProjectStar.user_id == user_id, ProjectStar.project_id == project.id
    )
    starred = session.exec(statement).first() is not None
    return ProjectPublic.from_orm(project, update={'starred': starred})
