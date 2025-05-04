from typing import Sequence

from sqlalchemy.orm import selectinload
from sqlmodel import Session, select

from app.models.project import Project


def insert(session: Session, project: Project) -> Project:
    session.add(project)
    session.commit()
    session.refresh(project)

    return project


def get_all(session: Session) -> Sequence[Project]:
    statement = select(Project).options(selectinload(Project.tags))
    return session.exec(statement).all()
