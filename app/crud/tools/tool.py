from collections.abc import Sequence

from fastapi import HTTPException
from sqlalchemy import func, text
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, select
from starlette.status import HTTP_404_NOT_FOUND

from app.models.tools import ToolDefine
from app.schemas.tools import DockerToolUpdate

_FUZZY_SIMILARITY_THRESHOLD = 0.1


def insert(session: Session, tool: ToolDefine) -> ToolDefine:
    """Insert a new tool into the database."""
    session.add(tool)
    session.commit()
    session.refresh(tool)

    return tool


def insert_all(session: Session, tools: list[ToolDefine]) -> list[ToolDefine]:
    """Insert multiple tools into the database."""
    session.add_all(tools)
    session.commit()

    for tool in tools:
        session.refresh(tool)

    return tools


def delete(session: Session, tool_uid: str) -> None:
    """Delete a tool by its UID."""
    tool = get(session, tool_uid)
    if tool:
        session.delete(tool)
        session.commit()


def drop_all(session: Session) -> int:
    """Truncate the tool table and return the number of deleted rows."""
    count_stmt = select(func.count()).select_from(ToolDefine)
    count = session.exec(count_stmt).one()

    if count == 0:
        return 0

    table_name = ToolDefine.__tablename__
    # pyrefly: ignore [no-matching-overload]
    session.exec(text(f'TRUNCATE TABLE {table_name} CASCADE'))
    session.commit()

    return count


def update(session: Session, tool: ToolDefine) -> ToolDefine:
    """Update an existing tool in the database."""
    session.add(tool)
    session.commit()
    session.refresh(tool)

    return tool


def update_partial(session: Session, tool_uid: str, tool_update: DockerToolUpdate) -> ToolDefine | None:
    """部分更新工具配置."""
    from app.crud.tools import tool_tag as tag_crud

    db_tool = get(session, tool_uid)
    if db_tool is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail='Tool not found')

    # 准备更新数据
    update_data = tool_update.model_dump(exclude_unset=True)
    db_tool.sqlmodel_update(update_data)

    # 处理标签更新
    if tool_update.tags is not None:
        # pyrefly: ignore [bad-assignment]
        exist_ids: list[int] = [tag.id for tag in tool_update.tags if tag.id is not None and tag.id != -1]
        tags = tag_crud.get_all_by_id(session, exist_ids)
        db_tool.tags = list(tags)

    session.add(db_tool)
    session.commit()
    session.refresh(db_tool)

    return db_tool


def get_page(session: Session, limit: int, offset: int) -> tuple[Sequence[ToolDefine], int]:
    """Return a page of tools together with the total count."""
    page_stmt = (
        select(ToolDefine)
        .order_by(col(ToolDefine.name))
        # pyrefly: ignore [bad-argument-type]
        .options(selectinload(ToolDefine.tags))
        .offset(offset)
        .limit(limit)
    )
    count_stmt = select(func.count()).select_from(ToolDefine)
    return session.exec(page_stmt).all(), session.exec(count_stmt).one()


def get_all_by_group(session: Session, group_id: int | None, limit: int, offset: int) -> Sequence[ToolDefine]:
    """Return a paginated list of tools belonging to the given group."""
    stmt = (
        select(ToolDefine)
        .where(ToolDefine.group_id == group_id)
        .options(selectinload(ToolDefine.tags))  # type: ignore
        .offset(offset)
        .limit(limit)
    )

    return session.exec(stmt).all()


def get_all_by_image(session: Session, image_uid: str) -> Sequence[ToolDefine]:
    """Return all tools associated with the given image UID."""
    stmt = select(ToolDefine).where(ToolDefine.image_uid == image_uid)

    return session.exec(stmt).all()


def get_all_by_uid(session: Session, uids: list[str]) -> Sequence[ToolDefine]:
    """Return all tools whose UIDs are in the given list."""
    stmt = select(ToolDefine).where(col(ToolDefine.uid).in_(uids))
    return session.exec(stmt).all()


def get_by_name_fuzzy(session: Session, name: str, limit: int, offset: int) -> Sequence[ToolDefine]:
    """Return tools whose names are similar to the given string, ordered by similarity."""
    stmt = (
        select(ToolDefine)
        .where(func.similarity(ToolDefine.name, name) > _FUZZY_SIMILARITY_THRESHOLD)
        .order_by(func.similarity(ToolDefine.name, name).desc())
        .options(selectinload(ToolDefine.tags))  # type: ignore
        .offset(offset)
        .limit(limit)
    )

    return session.exec(stmt).all()


def get_page_by_name_fuzzy(session: Session, name: str, limit: int, offset: int) -> tuple[Sequence[ToolDefine], int]:
    """Return a page of fuzzy-matched tools together with the total match count."""
    count_stmt = (
        select(func.count())
        .select_from(ToolDefine)
        .where(func.similarity(ToolDefine.name, name) > _FUZZY_SIMILARITY_THRESHOLD)
    )
    return get_by_name_fuzzy(session, name, limit, offset), session.exec(count_stmt).one()


def get(session: Session, tool_uid: str) -> ToolDefine | None:
    """Return a tool by its UID."""
    stmt = select(ToolDefine).where(ToolDefine.uid == tool_uid)
    return session.exec(stmt).one_or_none()
