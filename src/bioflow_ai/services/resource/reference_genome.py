"""Application services for reference-genome persistence."""

from sqlalchemy.orm import Session

from bioflow_ai.crud.resource import reference_genome as genome_crud
from bioflow_ai.models.resource import ReferenceGenomes
from bioflow_ai.schemas.resource import GenomeUpdate
from bioflow_ai.services._transaction import committing


class ReferenceGenomeServiceError(ValueError):
    """Base error for reference-genome application operations."""


class ReferenceGenomeNotFoundError(ReferenceGenomeServiceError):
    """Raised when a requested reference genome does not exist."""


def create_genome(session: Session, genome: ReferenceGenomes) -> ReferenceGenomes:
    """Create and commit a reference genome."""
    with committing(session):
        genome_crud.insert(session, genome)
    session.refresh(genome)
    return genome


def update_genome(session: Session, data: GenomeUpdate) -> ReferenceGenomes:
    """Update and commit a reference genome."""
    with committing(session):
        genome = genome_crud.update(session, data)
        if genome is None:
            raise ReferenceGenomeNotFoundError
    session.refresh(genome)
    return genome


def delete_genome(session: Session, genome_id: int) -> None:
    """Delete and commit a reference genome."""
    with committing(session):
        if not genome_crud.delete(session, genome_id):
            raise ReferenceGenomeNotFoundError
