from typing import Optional

from fastapi import HTTPException
from sqlalchemy import Select, func
from sqlmodel import Session, select, col
from starlette.status import HTTP_409_CONFLICT

from app.models.resource import BioDb


def insert(session: Session, bio_db: BioDb) -> BioDb:
    if get_by_name(session, bio_db.name) is not None:
        raise HTTPException(HTTP_409_CONFLICT, 'name already exist')

    session.add(bio_db)
    session.commit()
    session.refresh(bio_db)

    return bio_db


def delete(session: Session, db_id: int):
    bio_db = get(session, db_id)
    session.delete(bio_db)
    session.commit()


def get_counts(session: Session):
    query: Select = select(func.count()).select_from(BioDb)

    return session.exec(query).one()


def get_all(session: Session, limit: int, offset: int):
    query: Select = select(BioDb).order_by(col(BioDb.name).desc()).limit(limit).offset(offset)

    return session.exec(query).all()


def get_by_name(session: Session, name: str) -> Optional[BioDb]:
    query: Select = select(BioDb).where(BioDb.name == name)

    return session.exec(query).one_or_none()


def get_by_name_fuzzy(session: Session, name: str, limit: int, offset: int):
    query: Select = (
        select(BioDb)
        .where(func.similarity(BioDb.name, name) > 0.1)
        .order_by(func.similarity(BioDb.name, name).desc(), col(BioDb.name).desc())
        .offset(offset)
        .limit(limit)
    )

    return session.exec(query).all()


def get(session: Session, db_id: int):
    query: Select = select(BioDb).where(BioDb.id == db_id)
    return session.exec(query).one()
