from collections.abc import Sequence
from typing import Any

from sqlalchemy import and_, case, func, or_, select, text
from sqlalchemy.exc import DataError
from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy.sql.elements import ColumnElement

from bioflow_ai.models.project import Project
from bioflow_ai.models.tools import ToolDefine
from bioflow_ai.models.workflow import NodeRun, Workflow, WorkflowRun
from bioflow_ai.schemas.tools import ToolRunUsage, ToolWorkflowUsage
from bioflow_ai.utils.identifiers import UUIDLike, as_uuid

_FUZZY_SIMILARITY_THRESHOLD = 0.1
_PRIMARY_TERM_SIMILARITY_THRESHOLD = 0.3


def _name_search_expressions(
    name: str,
) -> tuple[ColumnElement[bool], tuple[Any, ...]]:
    """Build relevance filtering and ordering for a tool-name query.

    The first whitespace-delimited term is treated as the likely tool name. This
    keeps qualifiers useful for tie-breaking without allowing them to overwhelm
    a slightly misspelled tool name (for example, ``RTAR RNA-seq``).
    """
    normalized_name = ' '.join(name.split())
    primary_term = normalized_name.split(maxsplit=1)[0] if normalized_name else ''
    lower_name = func.lower(ToolDefine.name)
    full_similarity = func.similarity(ToolDefine.name, normalized_name)
    primary_similarity = func.word_similarity(primary_term, ToolDefine.name)

    match_priority = case(
        (lower_name == normalized_name.casefold(), 4),
        (lower_name.startswith(normalized_name.casefold(), autoescape=True), 3),
        (lower_name == primary_term.casefold(), 2),
        (lower_name.startswith(primary_term.casefold(), autoescape=True), 1),
        else_=0,
    )
    predicate = or_(
        full_similarity > _FUZZY_SIMILARITY_THRESHOLD,
        primary_similarity > _PRIMARY_TERM_SIMILARITY_THRESHOLD,
    )
    ordering = (
        match_priority.desc(),
        primary_similarity.desc(),
        full_similarity.desc(),
        ToolDefine.name,
    )
    return predicate, ordering


def insert(session: Session, tool: ToolDefine) -> ToolDefine:
    """Insert a new tool into the database."""
    session.add(tool)
    session.flush()
    return tool


def insert_all(session: Session, tools: list[ToolDefine]) -> list[ToolDefine]:
    """Insert multiple tools into the database."""
    session.add_all(tools)
    session.flush()
    return tools


def delete(session: Session, tool_uid: UUIDLike) -> bool:
    """Delete a tool by its UID and report whether it existed."""
    tool = get(session, tool_uid)
    if tool is None:
        return False
    session.delete(tool)
    session.flush()
    return True


def drop_all(session: Session) -> int:
    """Truncate the tool table and return the number of deleted rows."""
    count_stmt = select(func.count()).select_from(ToolDefine)
    count = session.scalar(count_stmt) or 0

    if count == 0:
        return 0

    table_name = ToolDefine.__tablename__
    session.execute(text(f'TRUNCATE TABLE {table_name} CASCADE'))

    return count


def update(session: Session, tool: ToolDefine) -> ToolDefine:
    """Update an existing tool in the database."""
    session.add(tool)
    session.flush()
    return tool


def get_page(session: Session, limit: int, offset: int) -> tuple[Sequence[ToolDefine], int]:
    """Return a page of tools together with the total count."""
    page_stmt = (
        select(ToolDefine)
        .order_by(ToolDefine.name)
        .options(selectinload(ToolDefine.tags), selectinload(ToolDefine.image))
        .offset(offset)
        .limit(limit)
    )
    count_stmt = select(func.count()).select_from(ToolDefine)
    return session.scalars(page_stmt).all(), session.scalar(count_stmt) or 0


def get_all_by_group(session: Session, group_id: int | None, limit: int, offset: int) -> Sequence[ToolDefine]:
    """Return a paginated list of tools belonging to the given group."""
    stmt = (
        select(ToolDefine)
        .where(ToolDefine.group_id == group_id)
        .options(selectinload(ToolDefine.tags), selectinload(ToolDefine.image))
        .offset(offset)
        .limit(limit)
    )

    return session.scalars(stmt).all()


def get_all_by_image(session: Session, image_uid: UUIDLike) -> Sequence[ToolDefine]:
    """Return all tools associated with the given image UID."""
    stmt = select(ToolDefine).where(ToolDefine.image_uid == as_uuid(image_uid))

    return session.scalars(stmt).all()


def get_all_by_uid(session: Session, uids: Sequence[UUIDLike], *, refresh: bool = False) -> Sequence[ToolDefine]:
    """Return tools in the same relevance order as the requested UIDs."""
    if not uids:
        return []

    normalized_uids = [as_uuid(uid) for uid in uids]
    stmt = (
        select(ToolDefine)
        .where(ToolDefine.uid.in_(normalized_uids))
        .options(selectinload(ToolDefine.tags))
        .execution_options(populate_existing=refresh)
    )
    tools_by_uid = {str(tool.uid): tool for tool in session.scalars(stmt).all()}
    requested_uids = dict.fromkeys(str(uid) for uid in normalized_uids)
    return [tools_by_uid[uid] for uid in requested_uids if uid in tools_by_uid]


def get_by_name_fuzzy(session: Session, name: str, limit: int, offset: int) -> Sequence[ToolDefine]:
    """Return tools ordered by exact, prefix, primary-term, and full-query relevance."""
    predicate, ordering = _name_search_expressions(name)
    stmt = (
        select(ToolDefine)
        .where(predicate)
        .order_by(*ordering)
        .options(selectinload(ToolDefine.tags), selectinload(ToolDefine.image))
        .offset(offset)
        .limit(limit)
    )

    return session.scalars(stmt).all()


def get_page_by_name_fuzzy(session: Session, name: str, limit: int, offset: int) -> tuple[Sequence[ToolDefine], int]:
    """Return a page of fuzzy-matched tools together with the total match count."""
    predicate, _ordering = _name_search_expressions(name)
    count_stmt = select(func.count()).select_from(ToolDefine).where(predicate)
    return get_by_name_fuzzy(session, name, limit, offset), session.scalar(count_stmt) or 0


def search_by_command(session: Session, patterns: Sequence[str], limit: int) -> tuple[Sequence[ToolDefine], int]:
    """Return tools whose rendered command matches every pattern, together with the total match count.

    Patterns are case-insensitive PostgreSQL regular expressions, similar to chaining ``rg -i``.
    If any pattern is not a valid regular expression, all patterns are matched as literal substrings.
    """
    try:
        with session.begin_nested():
            predicate = and_(*(ToolDefine.complete_command.regexp_match(pattern, flags='i') for pattern in patterns))
            return _search_by_predicate(session, predicate, limit)
    except DataError:
        predicate = and_(*(ToolDefine.complete_command.icontains(pattern, autoescape=True) for pattern in patterns))
        return _search_by_predicate(session, predicate, limit)


def _search_by_predicate(
    session: Session,
    predicate: ColumnElement[bool],
    limit: int,
) -> tuple[Sequence[ToolDefine], int]:
    """Return a name-ordered page of tools matching a predicate and the total match count."""
    total = session.scalar(select(func.count()).select_from(ToolDefine).where(predicate)) or 0
    stmt = (
        select(ToolDefine)
        .where(predicate)
        .order_by(ToolDefine.name)
        .options(selectinload(ToolDefine.tags), selectinload(ToolDefine.image))
        .limit(limit)
    )
    return session.scalars(stmt).all(), total


def get(session: Session, tool_uid: UUIDLike) -> ToolDefine | None:
    """Return a tool by its UID."""
    stmt = (
        select(ToolDefine)
        .where(ToolDefine.uid == as_uuid(tool_uid))
        .options(
            selectinload(ToolDefine.tags),
            selectinload(ToolDefine.image),
            selectinload(ToolDefine.help_doc),
        )
    )
    return session.scalar(stmt)


def _workflow_reference_predicate(tool_uid: UUIDLike) -> ColumnElement[bool]:
    """Build a JSONB containment predicate for tool nodes in saved workflows."""
    return Workflow.workflow.contains(
        {'nodes': [{'type': 'tool', 'data': {'tool_uid': str(as_uuid(tool_uid))}}]},
    )


def get_usage_counts(session: Session, tool_uid: UUIDLike) -> tuple[int, int]:
    """Return saved-workflow and distinct historical-run counts for a tool."""
    normalized_uid = as_uuid(tool_uid)
    workflow_count = (
        session.scalar(
            select(func.count()).select_from(Workflow).where(_workflow_reference_predicate(normalized_uid)),
        )
        or 0
    )
    run_count = (
        session.scalar(
            select(func.count(func.distinct(NodeRun.run_uid))).where(NodeRun.tool_uid == normalized_uid),
        )
        or 0
    )
    return workflow_count, run_count


def get_workflow_usage_page(
    session: Session,
    tool_uid: UUIDLike,
    limit: int,
    offset: int,
) -> list[ToolWorkflowUsage]:
    """Return a page of saved workflows that reference a tool."""
    normalized_uid = as_uuid(tool_uid)
    statement = (
        select(Workflow)
        .where(_workflow_reference_predicate(normalized_uid))
        .order_by(Workflow.name, Workflow.uid)
        .offset(offset)
        .limit(limit)
    )
    tool_uid_text = str(normalized_uid)
    return [
        ToolWorkflowUsage(
            uid=workflow.uid,
            name=workflow.name,
            description=workflow.description,
            public=workflow.public,
            node_count=sum(
                1
                for node in workflow.workflow.nodes
                if node.type == 'tool' and node.data.get('tool_uid') == tool_uid_text
            ),
        )
        for workflow in session.scalars(statement).all()
    ]


def get_run_usage_page(
    session: Session,
    tool_uid: UUIDLike,
    limit: int,
    offset: int,
) -> list[ToolRunUsage]:
    """Return a page of distinct runs that contain tasks for a tool."""
    normalized_uid = as_uuid(tool_uid)
    task_count = func.count(NodeRun.uid).label('task_count')
    statement = (
        select(
            WorkflowRun.uid,
            WorkflowRun.name,
            WorkflowRun.project_id,
            Project.name.label('project_name'),
            WorkflowRun.workflow_uid,
            Workflow.name.label('workflow_name'),
            WorkflowRun.status,
            WorkflowRun.create_time,
            task_count,
        )
        .join(NodeRun, NodeRun.run_uid == WorkflowRun.uid)
        .outerjoin(Project, Project.id == WorkflowRun.project_id)
        .outerjoin(Workflow, Workflow.uid == WorkflowRun.workflow_uid)
        .where(NodeRun.tool_uid == normalized_uid)
        .group_by(
            WorkflowRun.uid,
            WorkflowRun.name,
            WorkflowRun.project_id,
            Project.name,
            WorkflowRun.workflow_uid,
            Workflow.name,
            WorkflowRun.status,
            WorkflowRun.create_time,
        )
        .order_by(WorkflowRun.create_time.desc().nullslast(), WorkflowRun.uid)
        .offset(offset)
        .limit(limit)
    )
    return [ToolRunUsage.model_validate(row) for row in session.execute(statement).mappings().all()]


def get_summary(session: Session, tool_uid: UUIDLike) -> ToolDefine | None:
    """Load the tool row without catalog relationships."""
    return session.get(ToolDefine, as_uuid(tool_uid))


def get_with_image(session: Session, tool_uid: UUIDLike) -> ToolDefine | None:
    """Load a tool and its image for execution."""
    return session.scalar(
        select(ToolDefine).where(ToolDefine.uid == as_uuid(tool_uid)).options(joinedload(ToolDefine.image))
    )
