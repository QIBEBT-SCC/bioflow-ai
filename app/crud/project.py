from datetime import UTC, datetime

from fastapi import HTTPException
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, or_, select
from starlette.status import HTTP_404_NOT_FOUND

from app.models.project import Project, ProjectFileMapping, ProjectStar
from app.schemas.project import ProjectPublic, ProjectUpdate
from app.schemas.user import UserPublic


def _to_project_public(project: Project, *, starred: bool) -> ProjectPublic:
    """Convert a Project model to a ProjectPublic schema."""
    project_dict = project.model_dump()
    project_dict['starred'] = starred
    project_dict['tags'] = project.tags
    if project.owner:
        project_dict['owner'] = UserPublic.model_validate(project.owner.model_dump())
    return ProjectPublic.model_validate(project_dict)


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
    """Update a project."""
    raise NotImplementedError


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


def get_all(session: Session, limit: int, offset: int, user_id: int) -> list[ProjectPublic]:
    """Return a paginated list of public and owned projects."""
    statement = (
        select(Project)
        .where(or_(Project.public, Project.owner_id == user_id))
        .options(selectinload(Project.tags), selectinload(Project.owner))  # type: ignore
        .offset(offset)
        .limit(limit)
    )
    projects = session.exec(statement).all()

    project_ids = [p.id for p in projects]
    starred_ids = get_star_ids(session, user_id, project_ids)

    return [_to_project_public(project, starred=project.id in starred_ids) for project in projects]


def get_all_star(session: Session, limit: int, offset: int, user_id: int) -> list[ProjectPublic]:
    """Return a paginated list of projects starred by the user."""
    starred_ids = select(ProjectStar.project_id).where(ProjectStar.user_id == user_id).offset(offset).limit(limit)

    statement = (
        select(Project)
        .where(col(Project.id).in_(starred_ids))
        .options(selectinload(Project.tags), selectinload(Project.owner))  # type: ignore
    )

    projects = session.exec(statement).all()
    return [_to_project_public(project, starred=True) for project in projects]


def get_all_my(session: Session, limit: int, offset: int, user_id: int) -> list[ProjectPublic]:
    """Return a paginated list of projects owned by the user."""
    statement = (
        select(Project)
        .where(Project.owner_id == user_id)
        .options(selectinload(Project.tags), selectinload(Project.owner))  # type: ignore
        .offset(offset)
        .limit(limit)
    )
    projects = session.exec(statement).all()

    project_ids = [p.id for p in projects]
    starred_ids = get_star_ids(session, user_id, project_ids)

    return [_to_project_public(project, starred=project.id in starred_ids) for project in projects]


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


# ==================== Project File Mapping CRUD ====================


def create_file_mapping(
    session: Session, project_id: int, keyword: str, file_path: str, description: str = ''
) -> ProjectFileMapping:
    """Create a project file mapping."""
    mapping = ProjectFileMapping(project_id=project_id, keyword=keyword, file_path=file_path, description=description)
    session.add(mapping)
    session.commit()
    session.refresh(mapping)
    return mapping


def get_file_mapping(session: Session, project_id: int, keyword: str) -> ProjectFileMapping | None:
    """Return the file mapping for a given project and keyword."""
    statement = select(ProjectFileMapping).where(
        ProjectFileMapping.project_id == project_id, ProjectFileMapping.keyword == keyword
    )
    return session.exec(statement).one_or_none()


def get_all_file_mappings(session: Session, project_id: int) -> list[ProjectFileMapping]:
    """Return all file mappings for a project."""
    statement = select(ProjectFileMapping).where(ProjectFileMapping.project_id == project_id)
    return list(session.exec(statement).all())


def update_file_mapping(
    session: Session, mapping_id: int, file_path: str | None = None, description: str | None = None
) -> ProjectFileMapping:
    """Update a file mapping's path or description."""
    statement = select(ProjectFileMapping).where(ProjectFileMapping.id == mapping_id)
    mapping = session.exec(statement).one_or_none()

    if not mapping:
        raise HTTPException(HTTP_404_NOT_FOUND, 'File mapping not found')

    if file_path is not None:
        mapping.file_path = file_path
    if description is not None:
        mapping.description = description

    mapping.update_time = datetime.now(UTC)
    session.add(mapping)
    session.commit()
    session.refresh(mapping)
    return mapping


def delete_file_mapping(session: Session, mapping_id: int) -> None:
    """Delete a file mapping by ID."""
    statement = select(ProjectFileMapping).where(ProjectFileMapping.id == mapping_id)
    mapping = session.exec(statement).one_or_none()

    if not mapping:
        raise HTTPException(HTTP_404_NOT_FOUND, 'File mapping not found')

    session.delete(mapping)
    session.commit()
