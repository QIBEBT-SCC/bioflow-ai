"""Application services for persisted uploaded-file records."""

from sqlalchemy.orm import Session

from app.crud.resource import file as file_crud
from app.models.resource import File
from app.services._transaction import committing


def create_file(session: Session, file: File) -> File:
    """Create and commit an uploaded-file record."""
    with committing(session):
        file_crud.insert(session, file)
    session.refresh(file)
    return file
