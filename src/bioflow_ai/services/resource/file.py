"""Application services for persisted uploaded-file records."""

from sqlalchemy.orm import Session

from bioflow_ai.crud.resource import file as file_crud
from bioflow_ai.models.resource import File
from bioflow_ai.services._transaction import committing


def create_file(session: Session, file: File) -> File:
    """Create and commit an uploaded-file record."""
    with committing(session):
        file_crud.insert(session, file)
    session.refresh(file)
    return file
