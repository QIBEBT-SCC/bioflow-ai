from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.sql import Select
from sqlalchemy.sql.elements import ColumnElement

from bioflow_ai.models.project import Project, ProjectStar, ProjectTag
from bioflow_ai.schemas.project import ProjectPublic, ProjectSort, ProjectTagPublic, ProjectUpdate
from bioflow_ai.schemas.user import UserPublic
from bioflow_ai.utils.system import utc_now


def _to_project_public(project: Project, *, starred: bool) -> ProjectPublic:
    """Convert an eagerly loaded Project model to its API projection."""
    if project.owner is None:
        msg = f'Project {project.id} has no owner'
        raise ValueError(msg)
    return ProjectPublic(
        id=project.id,
        name=project.name,
        description=project.description,
        starred=starred,
        public=project.public,
        create_time=project.create_time,
        update_time=project.update_time,
        owner=UserPublic.model_validate(project.owner),
        tags=[ProjectTagPublic.model_validate(tag) for tag in project.tags],
    )


def _project_order(sort: ProjectSort) -> tuple[Any, Any]:
    """Return a stable database ordering for a project list."""
    if sort == 'nameAsc':
        return Project.name.asc(), Project.id.asc()
    if sort == 'nameDesc':
        return Project.name.desc(), Project.id.desc()
    return Project.update_time.desc(), Project.id.desc()


def _project_search_condition(search: str | None) -> ColumnElement[bool] | None:
    """Build a literal, case-insensitive substring match for project names."""
    keyword = search.strip() if search else ''
    if not keyword:
        return None
    return Project.name.icontains(keyword, autoescape=True)


def _with_project_filters(
    statement: Select[Any],
    search: str | None,
    tag_id: int | None,
) -> Select[Any]:
    """Apply project name and tag filters before ordering and pagination."""
    condition = _project_search_condition(search)
    if condition is not None:
        statement = statement.where(condition)
    if tag_id is not None:
        statement = statement.where(Project.tags.any(ProjectTag.id == tag_id))
    return statement


def insert(session: Session, project: Project) -> Project:
    """Stage a new project and populate database-generated values."""
    session.add(project)
    session.flush()
    session.refresh(project)
    return project


def delete(session: Session, project_id: int) -> bool:
    """Stage deletion of a project if it exists."""
    project = session.get(Project, project_id)
    if project is None:
        return False
    session.delete(project)
    session.flush()
    return True


def update(session: Session, project_id: int, data: ProjectUpdate) -> Project | None:
    """Stage editable project fields and tag replacement."""
    project = session.get(Project, project_id)
    if project is None:
        return None

    for field, value in data.model_dump(exclude_unset=True, exclude={'tags'}).items():
        setattr(project, field, value)

    if 'tags' in data.model_fields_set:
        tag_ids = [tag.id for tag in data.tags or [] if tag.id is not None]
        project.tags = list(session.scalars(select(ProjectTag).where(ProjectTag.id.in_(tag_ids)))) if tag_ids else []

    project.update_time = utc_now()
    session.flush()
    session.refresh(project)
    return project


def get_star_ids(session: Session, user_id: int, project_ids: list[int]) -> set[int]:
    """Return the set of project IDs starred by the given user."""
    if not project_ids:
        return set()
    statement = select(ProjectStar.project_id).where(
        ProjectStar.user_id == user_id,
        ProjectStar.project_id.in_(project_ids),
    )
    return set(session.scalars(statement))


def star_project(session: Session, user_id: int, project_id: int) -> bool:
    """Stage a project star if it does not already exist."""
    statement = select(ProjectStar).where(ProjectStar.user_id == user_id, ProjectStar.project_id == project_id)
    if session.scalar(statement) is not None:
        return False
    session.add(ProjectStar(user_id=user_id, project_id=project_id))
    session.flush()
    return True


def unstar_project(session: Session, user_id: int, project_id: int) -> bool:
    """Stage removal of a project star if it exists."""
    statement = select(ProjectStar).where(ProjectStar.user_id == user_id, ProjectStar.project_id == project_id)
    star = session.scalar(statement)
    if star is None:
        return False
    session.delete(star)
    session.flush()
    return True


def _list_projects(
    session: Session,
    statement: Select[tuple[Project]],
    *,
    user_id: int,
    starred: bool | None = None,
) -> list[ProjectPublic]:
    projects = list(session.scalars(statement).unique())
    if starred is True:
        starred_ids = {project.id for project in projects}
    elif starred is False:
        starred_ids = set()
    else:
        starred_ids = get_star_ids(session, user_id, [project.id for project in projects])
    return [_to_project_public(project, starred=project.id in starred_ids) for project in projects]


def get_all(
    session: Session,
    limit: int,
    offset: int,
    user_id: int,
    sort: ProjectSort = 'recent',
    search: str | None = None,
    tag_id: int | None = None,
    *,
    include_private: bool = False,
    include_owned: bool = True,
) -> list[ProjectPublic]:
    """Return a paginated list of public and owned projects."""
    statement = select(Project)
    if not include_private:
        visibility = or_(Project.public, Project.owner_id == user_id) if include_owned else Project.public
        statement = statement.where(visibility)
    statement = _with_project_filters(statement, search, tag_id)
    statement = (
        statement.options(selectinload(Project.tags), selectinload(Project.owner))
        .order_by(*_project_order(sort))
        .offset(offset)
        .limit(limit)
    )
    return _list_projects(session, statement, user_id=user_id)


def get_all_star(
    session: Session,
    limit: int,
    offset: int,
    user_id: int,
    sort: ProjectSort = 'recent',
    search: str | None = None,
    tag_id: int | None = None,
    *,
    include_private: bool = False,
    include_owned: bool = True,
) -> list[ProjectPublic]:
    """Return a paginated list of projects starred by the user."""
    statement = (
        select(Project).join(ProjectStar, ProjectStar.project_id == Project.id).where(ProjectStar.user_id == user_id)
    )
    if not include_private:
        visibility = or_(Project.public, Project.owner_id == user_id) if include_owned else Project.public
        statement = statement.where(visibility)
    statement = _with_project_filters(statement, search, tag_id)
    statement = (
        statement.options(selectinload(Project.tags), selectinload(Project.owner))
        .order_by(*_project_order(sort))
        .offset(offset)
        .limit(limit)
    )
    return _list_projects(session, statement, user_id=user_id, starred=True)


def get_all_my(
    session: Session,
    limit: int,
    offset: int,
    user_id: int,
    sort: ProjectSort = 'recent',
    search: str | None = None,
    tag_id: int | None = None,
) -> list[ProjectPublic]:
    """Return a paginated list of projects owned by the user."""
    statement = _with_project_filters(
        select(Project).where(Project.owner_id == user_id),
        search,
        tag_id,
    )
    statement = (
        statement.options(selectinload(Project.tags), selectinload(Project.owner))
        .order_by(*_project_order(sort))
        .offset(offset)
        .limit(limit)
    )
    return _list_projects(session, statement, user_id=user_id)


def get_page(
    session: Session,
    limit: int,
    offset: int,
    user_id: int,
    sort: ProjectSort = 'recent',
    search: str | None = None,
    tag_id: int | None = None,
    *,
    include_private: bool = False,
    include_owned: bool = True,
) -> tuple[list[ProjectPublic], int]:
    """Return a page of public and owned projects together with the total count."""
    count_statement = select(func.count()).select_from(Project)
    if not include_private:
        visibility = or_(Project.public, Project.owner_id == user_id) if include_owned else Project.public
        count_statement = count_statement.where(visibility)
    count_statement = _with_project_filters(count_statement, search, tag_id)
    projects = get_all(
        session,
        limit,
        offset,
        user_id,
        sort,
        search,
        tag_id,
        include_private=include_private,
        include_owned=include_owned,
    )
    return projects, session.scalar(count_statement) or 0


def get_page_star(
    session: Session,
    limit: int,
    offset: int,
    user_id: int,
    sort: ProjectSort = 'recent',
    search: str | None = None,
    tag_id: int | None = None,
    *,
    include_private: bool = False,
    include_owned: bool = True,
) -> tuple[list[ProjectPublic], int]:
    """Return a page of the user's starred projects together with the total count."""
    count_statement = (
        select(func.count())
        .select_from(Project)
        .join(ProjectStar, ProjectStar.project_id == Project.id)
        .where(ProjectStar.user_id == user_id)
    )
    if not include_private:
        visibility = or_(Project.public, Project.owner_id == user_id) if include_owned else Project.public
        count_statement = count_statement.where(visibility)
    count_statement = _with_project_filters(count_statement, search, tag_id)
    projects = get_all_star(
        session,
        limit,
        offset,
        user_id,
        sort,
        search,
        tag_id,
        include_private=include_private,
        include_owned=include_owned,
    )
    return projects, session.scalar(count_statement) or 0


def get_page_my(
    session: Session,
    limit: int,
    offset: int,
    user_id: int,
    sort: ProjectSort = 'recent',
    search: str | None = None,
    tag_id: int | None = None,
) -> tuple[list[ProjectPublic], int]:
    """Return a page of the user's owned projects together with the total count."""
    count_statement = select(func.count()).select_from(Project).where(Project.owner_id == user_id)
    count_statement = _with_project_filters(count_statement, search, tag_id)
    return get_all_my(session, limit, offset, user_id, sort, search, tag_id), session.scalar(count_statement) or 0


def get(session: Session, user_id: int, project_id: int) -> ProjectPublic | None:
    """Return a single project by ID, with star status for the user."""
    statement = (
        select(Project).where(Project.id == project_id).options(selectinload(Project.tags), selectinload(Project.owner))
    )
    project = session.scalar(statement)
    if project is None:
        return None
    starred_statement = select(ProjectStar.project_id).where(
        ProjectStar.user_id == user_id,
        ProjectStar.project_id == project.id,
    )
    return _to_project_public(project, starred=session.scalar(starred_statement) is not None)


def get_recent(session: Session, user_id: int, *, include_owned: bool = True) -> ProjectPublic | None:
    """Return the most recently updated project owned by the user."""
    statement = (
        select(Project)
        .where(Project.owner_id == user_id if include_owned else Project.public)
        .order_by(Project.update_time.desc())
        .options(selectinload(Project.tags), selectinload(Project.owner))
    )
    project = session.scalar(statement)
    if project is None:
        return None
    starred_statement = select(ProjectStar.project_id).where(
        ProjectStar.user_id == user_id,
        ProjectStar.project_id == project.id,
    )
    return _to_project_public(project, starred=session.scalar(starred_statement) is not None)


def get_by_id(session: Session, project_id: int) -> Project | None:
    """Load a project record for service-level authorization and orchestration."""
    return session.get(Project, project_id)
