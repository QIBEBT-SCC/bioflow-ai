"""Application services for persisted uploaded-file records."""

import hashlib

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from bioflow_ai.crud.resource import file as file_crud
from bioflow_ai.infrastructure.object_storage.files import write_bytes
from bioflow_ai.models.resource import File
from bioflow_ai.services._transaction import committing


def store_file_bytes(session: Session, data: bytes, filename: str, media_type: str, prefix: str) -> File:
    """Store immutable content once in the existing globally deduplicated file registry."""
    digest = hashlib.sha256(data).hexdigest()
    file = file_crud.get_by_sha256(session, digest)
    if file is None:
        key = f'{prefix}/{digest}'
        write_bytes(key, data, media_type)
        try:
            with session.begin_nested():
                file = file_crud.insert(session, File(sha256=digest, file_name=filename, s3_path=key))
        except IntegrityError:
            file = session.scalar(select(File).where(File.sha256 == digest))
            if file is None:
                raise
    return file


def create_file(session: Session, file: File) -> File:
    """Create and commit an uploaded-file record."""
    with committing(session):
        file_crud.insert(session, file)
    session.refresh(file)
    return file
