from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, func, or_, select
from starlette.status import HTTP_404_NOT_FOUND

from app.models.project import Project, ProjectStar, ProjectTag
from app.schemas.project import ProjectPublic, ProjectSort, ProjectUpdate
from app.schemas.user import UserPublic
from app.utils.system import utc_now


def _to_project_public(project: Project, *, starred: bool) -> ProjectPublic:
    """Convert a Project model to a ProjectPublic schema."""
    project_dict = project.model_dump()
    project_dict['starred'] = starred
    project_dict['tags'] = project.tags
    if project.owner:
        project_dict['owner'] = UserPublic.model_validate(project.owner)
    return ProjectPublic.model_validate(project_dict)


def _project_order(sort: ProjectSort) -> tuple[Any, Any]:
    """Return a stable database ordering for a project list."""
    if sort == 'nameAsc':
        return col(Project.name).asc(), col(Project.id).asc()
    if sort == 'nameDesc':
        return col(Project.name).desc(), col(Project.id).desc()
    return col(Project.update_time).desc(), col(Project.id).desc()


def insert(session: Session, project: Project) -> Project:
    """Insert a new project into the database."""
    session.add(project)
    session.commit()
    session.refresh(project)

    return project


def delete(session: Session, project_id: int) -> None:
    """Delete a project by ID if it exists."""
    statement = select(Project).where(Project.id == project_id)
    project = session.exec(statement).one_or_none()

    if project:
        session.delete(project)
        session.commit()


def update(session: Session, project_id: int, project: ProjectUpdate) -> Project:
    """Update a project's editable fields and replace its tags when supplied."""
    db_project = session.get(Project, project_id)
    if db_project is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Project not found')

    update_data = project.model_dump(exclude_unset=True, exclude={'tags'})
    for field, value in update_data.items():
        setattr(db_project, field, value)

    if 'tags' in project.model_fields_set:
        tag_ids = [tag.id for tag in project.tags or [] if tag.id is not None]
        tags = session.exec(select(ProjectTag).where(col(ProjectTag.id).in_(tag_ids))).all() if tag_ids else []
        db_project.tags = list(tags)

    db_project.update_time = utc_now()
    session.add(db_project)
    session.commit()
    session.refresh(db_project)
    return db_project


def get_star_ids(session: Session, user_id: int, project_ids: list[int]) -> set[int]:
    """Return the set of project IDs starred by the given user."""
    star_stmt = select(ProjectStar.project_id).where(
        ProjectStar.user_id == user_id, col(ProjectStar.project_id).in_(project_ids)
    )
    return set(session.exec(star_stmt).all())


def star_project(session: Session, user_id: int, project_id: int) -> None:
    """Star a project for a user if not already starred."""
    statement = select(ProjectStar).where(ProjectStar.user_id == user_id, ProjectStar.project_id == project_id)
    exists = session.exec(statement).first()
    if not exists:
        star = ProjectStar(user_id=user_id, project_id=project_id)
        session.add(star)
        session.commit()


def unstar_project(session: Session, user_id: int, project_id: int) -> None:
    """Remove a star from a project for a user."""
    statement = select(ProjectStar).where(ProjectStar.user_id == user_id, ProjectStar.project_id == project_id)
    star = session.exec(statement).first()
    if star:
        session.delete(star)
        session.commit()


def get_all(
    session: Session, limit: int, offset: int, user_id: int, sort: ProjectSort = 'recent'
) -> list[ProjectPublic]:
    """Return a paginated list of public and owned projects."""
    statement = (
        select(Project)
        .where(or_(Project.public, Project.owner_id == user_id))
        .options(selectinload(Project.tags), selectinload(Project.owner))  # type: ignore
        .order_by(*_project_order(sort))
        .offset(offset)
        .limit(limit)
    )
    projects = session.exec(statement).all()

    project_ids = [p.id for p in projects]
    starred_ids = get_star_ids(session, user_id, project_ids)

    return [_to_project_public(project, starred=project.id in starred_ids) for project in projects]


def get_all_star(
    session: Session, limit: int, offset: int, user_id: int, sort: ProjectSort = 'recent'
) -> list[ProjectPublic]:
    """Return a paginated list of projects starred by the user."""
    statement = (
        select(Project)
        # pyrefly: ignore [bad-argument-type]
        .join(ProjectStar, ProjectStar.project_id == Project.id)
        .where(ProjectStar.user_id == user_id)
        .options(selectinload(Project.tags), selectinload(Project.owner))  # type: ignore
        .order_by(*_project_order(sort))
        .offset(offset)
        .limit(limit)
    )

    projects = session.exec(statement).all()
    return [_to_project_public(project, starred=True) for project in projects]


def get_all_my(
    session: Session, limit: int, offset: int, user_id: int, sort: ProjectSort = 'recent'
) -> list[ProjectPublic]:
    """Return a paginated list of projects owned by the user."""
    statement = (
        select(Project)
        .where(Project.owner_id == user_id)
        .options(selectinload(Project.tags), selectinload(Project.owner))  # type: ignore
        .order_by(*_project_order(sort))
        .offset(offset)
        .limit(limit)
    )
    projects = session.exec(statement).all()

    project_ids = [p.id for p in projects]
    starred_ids = get_star_ids(session, user_id, project_ids)

    return [_to_project_public(project, starred=project.id in starred_ids) for project in projects]


def get_page(
    session: Session, limit: int, offset: int, user_id: int, sort: ProjectSort = 'recent'
) -> tuple[list[ProjectPublic], int]:
    """Return a page of public and owned projects together with the total count."""
    count_stmt = select(func.count()).select_from(Project).where(or_(Project.public, Project.owner_id == user_id))
    return get_all(session, limit, offset, user_id, sort), session.exec(count_stmt).one()


def get_page_star(
    session: Session, limit: int, offset: int, user_id: int, sort: ProjectSort = 'recent'
) -> tuple[list[ProjectPublic], int]:
    """Return a page of the user's starred projects together with the total count."""
    count_stmt = select(func.count()).select_from(ProjectStar).where(ProjectStar.user_id == user_id)
    return get_all_star(session, limit, offset, user_id, sort), session.exec(count_stmt).one()


def get_page_my(
    session: Session, limit: int, offset: int, user_id: int, sort: ProjectSort = 'recent'
) -> tuple[list[ProjectPublic], int]:
    """Return a page of the user's owned projects together with the total count."""
    count_stmt = select(func.count()).select_from(Project).where(Project.owner_id == user_id)
    return get_all_my(session, limit, offset, user_id, sort), session.exec(count_stmt).one()


def get(session: Session, user_id: int, project_id: int) -> ProjectPublic:
    """Return a single project by ID, with star status for the user."""
    statement = (
        select(Project).where(Project.id == project_id).options(selectinload(Project.tags), selectinload(Project.owner))  # type: ignore
    )
    project: Project = session.exec(statement).one()

    statement = select(ProjectStar).where(ProjectStar.user_id == user_id, ProjectStar.project_id == project.id)
    starred = session.exec(statement).first() is not None
    return _to_project_public(project, starred=starred)


def get_recent(session: Session, user_id: int) -> ProjectPublic | None:
    """Return the most recently updated project owned by the user."""
    statement = (
        select(Project)
        .where(Project.owner_id == user_id)
        .order_by(col(Project.update_time).desc())
        .options(selectinload(Project.tags), selectinload(Project.owner))  # type: ignore
    )
    project = session.exec(statement).first()

    if project is None:
        return None

    statement = select(ProjectStar).where(ProjectStar.user_id == user_id, ProjectStar.project_id == project.id)
    starred = session.exec(statement).first() is not None
    return _to_project_public(project, starred=starred)
