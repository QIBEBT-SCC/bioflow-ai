from typing import Optional

from sqlmodel import Session, select

from app.models.resource import File


def insert(session: Session, file: File):
    session.add(file)
    session.commit()
    session.refresh(file)
    return file



def get_by_sha256(session: Session, sha256: str) -> Optional[File]:
    statement = select(File).where(File.sha256 == sha256)

    return session.exec(statement).one_or_none()
