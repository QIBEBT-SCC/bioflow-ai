from sqlalchemy import Select
from sqlmodel import Session, select

from app.models.workflow import RunInstance
from app.schemas.run_instance import RunUpdate


def insert(session: Session, run: RunInstance) -> RunInstance:
    session.add(run)
    session.commit()
    session.refresh(run)

    return run


def update(session: Session, uid: str, run_update: RunUpdate) -> RunInstance:
    db_run = get(session, uid)

    db_run.sqlmodel_update(run_update.model_dump(exclude_unset=True))
    session.add(db_run)
    session.commit()
    session.refresh(db_run)
    return db_run


def get(session: Session, uid: str) -> RunInstance:
    statement: Select = select(RunInstance).where(RunInstance.uid == uid)
    return session.exec(statement).one()
