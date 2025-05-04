from sqlmodel import Session

from app.models.project import Project


def insert(session: Session, project: Project) -> Project:
    session.add(project)
    session.commit()
    session.refresh(project)

    return project
