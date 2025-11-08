from typing import Optional

from sqlalchemy import Select
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, or_, select

from app.models.project import Project, ProjectStar
from app.schemas.project import ProjectPublic, ProjectUpdate
from app.schemas.user import UserPublic


def _to_project_public(project: Project, starred: bool) -> ProjectPublic:
    """将 Project 模型转换为 ProjectPublic schema，包含 starred 字段"""
    project_dict = project.model_dump()
    project_dict['starred'] = starred
    project_dict['tags'] = project.tags
    if project.owner:
        project_dict['owner'] = UserPublic.model_validate(project.owner.model_dump())
    return ProjectPublic.model_validate(project_dict)


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
    statement: Select = select(ProjectStar).where(ProjectStar.user_id == user_id, ProjectStar.project_id == project_id)
    exists = session.exec(statement).first()
    if not exists:
        star = ProjectStar(user_id=user_id, project_id=project_id)
        session.add(star)
        session.commit()


def unstar_project(session: Session, user_id: int, project_id: int):
    statement: Select = select(ProjectStar).where(ProjectStar.user_id == user_id, ProjectStar.project_id == project_id)
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

    return [_to_project_public(project, project.id in starred_ids) for project in projects]


def get_all_star(session: Session, limit: int, offset: int, user_id: int) -> list[ProjectPublic]:
    starred_ids = select(ProjectStar.project_id).where(ProjectStar.user_id == user_id).offset(offset).limit(limit)

    statement: Select = (
        select(Project)
        .where(col(Project.id).in_(starred_ids))
        .options(selectinload(Project.tags), selectinload(Project.owner))  # type: ignore
    )

    projects = session.exec(statement).all()
    return [_to_project_public(project, starred=True) for project in projects]


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

    return [_to_project_public(project, project.id in starred_ids) for project in projects]


def get(session: Session, user_id: int, project_id: int) -> ProjectPublic:
    statement: Select = (
        select(Project).where(Project.id == project_id).options(selectinload(Project.tags), selectinload(Project.owner))  # type: ignore
    )
    project: Project = session.exec(statement).one()

    statement: Select = select(ProjectStar).where(ProjectStar.user_id == user_id, ProjectStar.project_id == project.id)
    starred = session.exec(statement).first() is not None
    return _to_project_public(project, starred)


def get_recent(session: Session, user_id: int) -> Optional[ProjectPublic]:
    statement: Select = (
        select(Project)
        .where(Project.owner_id == user_id)
        .order_by(col(Project.update_time).desc())
        .options(selectinload(Project.tags), selectinload(Project.owner))  # type: ignore
    )
    project: Project = session.exec(statement).first()

    if project is None:
        return None

    statement: Select = select(ProjectStar).where(ProjectStar.user_id == user_id, ProjectStar.project_id == project.id)
    starred = session.exec(statement).first() is not None
    return _to_project_public(project, starred)
