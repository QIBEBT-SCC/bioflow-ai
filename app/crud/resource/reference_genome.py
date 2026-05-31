from sqlalchemy import func, or_
from sqlmodel import Session, select

from app.models.resource import ReferenceGenomes
from app.schemas.resource import GenomeUpdate


def insert(session: Session, genome: ReferenceGenomes) -> ReferenceGenomes:
    """Insertion of a new reference genome."""
    session.add(genome)
    session.commit()
    session.refresh(genome)
    return genome


def delete(session: Session, genome_id: int) -> bool:
    """Remove reference genome."""
    genome = get(session, genome_id)
    if genome:
        session.delete(genome)
        session.commit()
        return True
    return False


def update(session: Session, genome_update: GenomeUpdate) -> ReferenceGenomes:
    """Updating reference genome information."""
    db_genome = get(session, genome_update.id)
    if not db_genome:
        msg = f'Genome id {genome_update.id} not found'
        raise ValueError(msg)

    update_data = genome_update.model_dump(exclude_unset=True)
    db_genome.sqlmodel_update(update_data)

    session.add(db_genome)
    session.commit()
    session.refresh(db_genome)
    return db_genome


def get(session: Session, genome_id: int) -> ReferenceGenomes | None:
    """Obtain the reference genome by ID."""
    query = select(ReferenceGenomes).where(ReferenceGenomes.id == genome_id)
    return session.exec(query).one_or_none()


def get_by_taxid(session: Session, tax_id: int) -> ReferenceGenomes | None:
    """Retrieve the reference genome by species ID."""
    query = select(ReferenceGenomes).where(ReferenceGenomes.ncbi_tax_id == tax_id)
    return session.exec(query).one_or_none()


def search(session: Session, search_term: str, threshold: float = 0.3) -> list[ReferenceGenomes]:
    """Search genomes by name or alias using fuzzy similarity matching.

    Args:
        session: Database session.
        search_term: Search keyword.
        threshold: Similarity threshold (0.0-1.0), default 0.3.

    Returns:
        Matching genomes ordered by similarity descending.
    """
    query = (
        select(ReferenceGenomes)
        .where(
            or_(
                func.similarity(ReferenceGenomes.name, search_term) > threshold,
                func.similarity(ReferenceGenomes.aliases, search_term) > threshold,
            )
        )
        .order_by(
            func.greatest(
                func.similarity(ReferenceGenomes.name, search_term),
                func.similarity(ReferenceGenomes.aliases, search_term),
            ).desc()
        )
    )

    return list(session.exec(query).all())


def get_count(session: Session) -> int:
    """Obtain the total number of reference genomes."""
    return session.exec(select(func.count()).select_from(ReferenceGenomes)).one()


def get_all(session: Session, limit: int = 20, offset: int = 0) -> list[ReferenceGenomes]:
    """分页获取参考基因组列表."""
    query = select(ReferenceGenomes).offset(offset).limit(limit)
    return list(session.exec(query).all())
