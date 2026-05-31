from sqlmodel import Session, select

from app.models.resource import File


def insert(session: Session, file: File) -> File:
    """Insert a new file record into the database."""
    session.add(file)
    session.commit()
    session.refresh(file)
    return file


def get_by_sha256(session: Session, sha256: str) -> File | None:
    """Return a file record matching the given SHA-256 hash, or None."""
    statement = select(File).where(File.sha256 == sha256)

    return session.exec(statement).one_or_none()
