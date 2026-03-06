from typing import Optional

from sqlalchemy import Select, func, or_
from sqlmodel import Session, select

from app.models.resource import ReferenceGenomes
from app.schemas.resource import GenomeUpdate


def insert(session: Session, genome: ReferenceGenomes) -> ReferenceGenomes:
    """插入新的参考基因组"""
    session.add(genome)
    session.commit()
    session.refresh(genome)
    return genome


def delete(session: Session, genome_id: int) -> bool:
    """删除参考基因组"""
    genome = get(session, genome_id)
    if genome:
        session.delete(genome)
        session.commit()
        return True
    return False


def update(session: Session, genome_update: GenomeUpdate) -> ReferenceGenomes:
    """更新参考基因组信息"""
    db_genome = get(session, genome_update.id)
    if not db_genome:
        raise ValueError(f'Genome id {genome_update.id} not found')

    update_data = genome_update.model_dump(exclude_unset=True)
    db_genome.sqlmodel_update(update_data)

    session.add(db_genome)
    session.commit()
    session.refresh(db_genome)
    return db_genome


def get(session: Session, genome_id: int) -> Optional[ReferenceGenomes]:
    """通过ID获取参考基因组"""
    query: Select = select(ReferenceGenomes).where(ReferenceGenomes.id == genome_id)
    return session.exec(query).one_or_none()


def get_by_taxid(session: Session, tax_id: int) -> Optional[ReferenceGenomes]:
    """通过物种id获取参考基因组"""
    query: Select = select(ReferenceGenomes).where(ReferenceGenomes.ncbi_tax_id == tax_id)
    return session.exec(query).one_or_none()


def search(session: Session, search_term: str, threshold: float = 0.3) -> list[ReferenceGenomes]:
    """
    搜索参考基因组，支持通过标准名称或别名（用|分隔）进行相似度匹配

    Args:
        session: 数据库会话
        search_term: 搜索词
        threshold: 相似度阈值 (0.0-1.0)，默认0.3

    Returns:
        匹配的参考基因组列表，按相似度降序排列
    """
    query: Select = (
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


def get_all(session: Session) -> list[ReferenceGenomes]:
    """获取所有参考基因组"""
    query: Select = select(ReferenceGenomes)
    return list(session.exec(query).all())


def get_count(session: Session) -> int:
    """获取参考基因组总数"""
    return session.exec(select(func.count()).select_from(ReferenceGenomes)).one()


def get_all_paginated(session: Session, limit: int = 20, offset: int = 0) -> list[ReferenceGenomes]:
    """分页获取参考基因组列表"""
    query: Select = select(ReferenceGenomes).offset(offset).limit(limit)
    return list(session.exec(query).all())
