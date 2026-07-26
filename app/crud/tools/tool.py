from collections.abc import Sequence
from typing import Any

from sqlalchemy import case, func, or_, select, text
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.sql.elements import ColumnElement

from app.models.tools import ToolDefine
from app.utils.identifiers import UUIDLike, as_uuid

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
    session.refresh(tool)

    return tool


def insert_all(session: Session, tools: list[ToolDefine]) -> list[ToolDefine]:
    """Insert multiple tools into the database."""
    session.add_all(tools)
    session.flush()

    for tool in tools:
        session.refresh(tool)

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
    session.refresh(tool)

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


def get_all_by_uid(session: Session, uids: list[UUIDLike]) -> Sequence[ToolDefine]:
    """Return tools in the same relevance order as the requested UIDs."""
    if not uids:
        return []

    normalized_uids = [as_uuid(uid) for uid in uids]
    stmt = select(ToolDefine).where(ToolDefine.uid.in_(normalized_uids)).options(selectinload(ToolDefine.tags))
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
