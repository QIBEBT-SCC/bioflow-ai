from collections.abc import Sequence

from sqlalchemy import ColumnElement, func, select
from sqlalchemy.orm import Session

from bioflow_ai.models.code import CodeDefine
from bioflow_ai.schemas.code import CodeUpdate
from bioflow_ai.utils.identifiers import UUIDLike, as_uuid


def insert(session: Session, code: CodeDefine) -> CodeDefine:
    """Stage a reusable CODE definition for insertion."""
    session.add(code)
    session.flush()
    session.refresh(code)
    return code


def update(session: Session, code_uid: UUIDLike, data: CodeUpdate) -> CodeDefine | None:
    """Stage updates to a reusable CODE definition."""
    code = get(session, code_uid)
    if code is None:
        return None

    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(code, field, value)

    session.flush()
    session.refresh(code)
    return code


def delete(session: Session, code_uid: UUIDLike) -> bool:
    """Stage deletion of a reusable CODE definition."""
    code = get(session, code_uid)
    if code is None:
        return False

    session.delete(code)
    session.flush()
    return True


def get(session: Session, code_uid: UUIDLike) -> CodeDefine | None:
    """Return a reusable CODE definition by UID."""
    return session.get(CodeDefine, as_uuid(code_uid))


def get_page(
    session: Session,
    limit: int,
    offset: int,
    *,
    query: str | None = None,
    node_type: str | None = None,
) -> tuple[Sequence[CodeDefine], int]:
    """Return a filtered page of reusable CODE definitions and the total count."""
    filters: list[ColumnElement[bool]] = []
    if normalized_query := (query or '').strip():
        pattern = f'%{normalized_query}%'
        filters.append(CodeDefine.name.ilike(pattern))
    if node_type is not None:
        filters.append(CodeDefine.node_type == node_type)

    page_stmt = select(CodeDefine).where(*filters).order_by(CodeDefine.name).offset(offset).limit(limit)
    count_stmt = select(func.count()).select_from(CodeDefine).where(*filters)
    return session.scalars(page_stmt).all(), session.scalar(count_stmt) or 0


def get_all_by_uid(session: Session, uids: list[UUIDLike]) -> Sequence[CodeDefine]:
    """Return CODE definitions in the same order as the requested UIDs."""
    if not uids:
        return []

    normalized_uids = [as_uuid(uid) for uid in uids]
    stmt = select(CodeDefine).where(CodeDefine.uid.in_(normalized_uids))
    codes_by_uid = {code.uid: code for code in session.scalars(stmt).all()}
    return [codes_by_uid[uid] for uid in dict.fromkeys(normalized_uids) if uid in codes_by_uid]
