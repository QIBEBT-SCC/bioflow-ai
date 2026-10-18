from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from bioflow_ai.models.resource import BioDb

_FUZZY_SIMILARITY_THRESHOLD = 0.1


def insert(session: Session, bio_db: BioDb) -> BioDb:
    """Stage a new biological database entry."""
    session.add(bio_db)
    session.flush()
    session.refresh(bio_db)
    return bio_db


def delete(session: Session, db_id: int) -> bool:
    """Stage deletion of a biological database entry."""
    bio_db = get(session, db_id)
    if bio_db is None:
        return False
    session.delete(bio_db)
    session.flush()
    return True


def update(session: Session, db_id: int, changes: Mapping[str, Any]) -> BioDb | None:
    """Stage field updates to a biological database entry."""
    bio_db = get(session, db_id)
    if bio_db is None:
        return None
    for field, value in changes.items():
        setattr(bio_db, field, value)
    session.flush()
    session.refresh(bio_db)
    return bio_db


def update_download_result(session: Session, db_id: int, *, path: str, last_update: str) -> BioDb | None:
    """Stage a database-path update after a download task completes."""
    bio_db = get(session, db_id)
    if bio_db is None:
        return None
    bio_db.path = path
    bio_db.size = None
    bio_db.last_update = last_update
    session.flush()
    session.refresh(bio_db)
    return bio_db


def clear_download_result(session: Session, db_id: int) -> BioDb | None:
    """Stage clearing of a stored download result."""
    bio_db = get(session, db_id)
    if bio_db is None:
        return None
    bio_db.path = None
    bio_db.size = None
    bio_db.last_update = None
    session.flush()
    session.refresh(bio_db)
    return bio_db


def get_all(session: Session, limit: int, offset: int) -> Sequence[BioDb]:
    """Get a paginated list of all biological database entries."""
    query = select(BioDb).order_by(BioDb.name.desc()).limit(limit).offset(offset)
    return session.scalars(query).all()


def get_page(session: Session, limit: int, offset: int) -> tuple[Sequence[BioDb], int]:
    """Return a page of biological database entries together with the total count."""
    page_stmt = select(BioDb).order_by(BioDb.name.desc()).limit(limit).offset(offset)
    count_stmt = select(func.count()).select_from(BioDb)
    return session.scalars(page_stmt).all(), session.scalar(count_stmt) or 0


def get_by_name(session: Session, name: str) -> BioDb | None:
    """Get a biological database entry by exact name match."""
    query = select(BioDb).where(BioDb.name == name)

    return session.scalar(query)


def get_by_name_fuzzy(session: Session, name: str, limit: int, offset: int) -> Sequence[BioDb]:
    """Get biological database entries by fuzzy name or description match."""
    query = (
        select(BioDb)
        .where(
            or_(
                func.similarity(BioDb.name, name) > _FUZZY_SIMILARITY_THRESHOLD,
                func.similarity(BioDb.description, name) > _FUZZY_SIMILARITY_THRESHOLD,
            )
        )
        .order_by(
            func.greatest(func.similarity(BioDb.name, name), func.similarity(BioDb.description, name)),
            BioDb.name.desc(),
        )
        .offset(offset)
        .limit(limit)
    )

    return session.scalars(query).all()


def get_page_by_name_fuzzy(session: Session, name: str, limit: int, offset: int) -> tuple[Sequence[BioDb], int]:
    """Return a page of fuzzy-matched tools together with the total match count."""
    count_stmt = (
        select(func.count())
        .select_from(BioDb)
        .where(
            or_(
                func.similarity(BioDb.name, name) > _FUZZY_SIMILARITY_THRESHOLD,
                func.similarity(BioDb.description, name) > _FUZZY_SIMILARITY_THRESHOLD,
            )
        )
    )
    return get_by_name_fuzzy(session, name, limit, offset), session.scalar(count_stmt) or 0


def get(session: Session, db_id: int) -> BioDb | None:
    """Return a biological database entry by ID, or ``None``."""
    return session.get(BioDb, db_id)
