from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.resource import File


def insert(session: Session, file: File) -> File:
    """Stage a new file record."""
    session.add(file)
    session.flush()
    session.refresh(file)
    return file


def get_by_sha256(session: Session, sha256: str) -> File | None:
    """Return a file record matching the given SHA-256 hash, or None."""
    statement = select(File).where(File.sha256 == sha256)

    return session.scalar(statement)
