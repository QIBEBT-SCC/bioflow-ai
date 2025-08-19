from typing import Optional

from sqlalchemy import Select
from sqlmodel import Session, select

from app.models.resource import ReferenceGenomes


def insert(session: Session, genome: ReferenceGenomes) -> ReferenceGenomes:
    session.add(genome)
    session.commit()
    session.refresh(genome)

    return genome


def get(session: Session, genome_id: int) -> Optional[ReferenceGenomes]:
    query: Select = select(ReferenceGenomes).where(ReferenceGenomes.id == genome_id)

    return session.exec(query).one_or_none()
