from collections.abc import Sequence

from fastapi import HTTPException
from sqlalchemy import func
from sqlmodel import Session, col, or_, select
from starlette.status import HTTP_409_CONFLICT

from app.models.resource import BioDb

_FUZZY_SIMILARITY_THRESHOLD = 0.1


def insert(session: Session, bio_db: BioDb) -> BioDb:
    """Insert a new biological database entry."""
    if get_by_name(session, bio_db.name) is not None:
        raise HTTPException(HTTP_409_CONFLICT, 'name already exist')

    session.add(bio_db)
    session.commit()
    session.refresh(bio_db)

    return bio_db


def delete(session: Session, db_id: int) -> None:
    """Delete a biological database entry by ID."""
    bio_db = get(session, db_id)
    session.delete(bio_db)
    session.commit()


def update_download_result(session: Session, db_id: int, *, path: str, last_update: str) -> BioDb:
    """Update the database path after a download task completes."""
    bio_db = get(session, db_id)
    bio_db.path = path
    bio_db.size = None
    bio_db.last_update = last_update
    session.add(bio_db)
    session.commit()
    session.refresh(bio_db)
    return bio_db


def clear_download_result(session: Session, db_id: int) -> BioDb:
    """Clear stored download result before a fresh download attempt."""
    bio_db = get(session, db_id)
    bio_db.path = None
    bio_db.size = None
    bio_db.last_update = None
    session.add(bio_db)
    session.commit()
    session.refresh(bio_db)
    return bio_db


def get_all(session: Session, limit: int, offset: int) -> Sequence[BioDb]:
    """Get a paginated list of all biological database entries."""
    query = select(BioDb).order_by(col(BioDb.name).desc()).limit(limit).offset(offset)

    return session.exec(query).all()


def get_page(session: Session, limit: int, offset: int) -> tuple[Sequence[BioDb], int]:
    """Return a page of biological database entries together with the total count."""
    page_stmt = select(BioDb).order_by(col(BioDb.name).desc()).limit(limit).offset(offset)
    count_stmt = select(func.count()).select_from(BioDb)
    return session.exec(page_stmt).all(), session.exec(count_stmt).one()


def get_by_name(session: Session, name: str) -> BioDb | None:
    """Get a biological database entry by exact name match."""
    query = select(BioDb).where(BioDb.name == name)

    return session.exec(query).one_or_none()


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
            col(BioDb.name).desc(),
        )
        .offset(offset)
        .limit(limit)
    )

    return session.exec(query).all()


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
    return get_by_name_fuzzy(session, name, limit, offset), session.exec(count_stmt).one()


def get(session: Session, db_id: int) -> BioDb:
    """Get a biological database entry by its ID."""
    query = select(BioDb).where(BioDb.id == db_id)
    return session.exec(query).one()
