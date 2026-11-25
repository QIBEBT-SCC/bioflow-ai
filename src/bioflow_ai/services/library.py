from sqlalchemy import func, select
from sqlalchemy.orm import Session

from bioflow_ai.models.code import CodeDefine
from bioflow_ai.models.resource import BioDb, ReferenceGenomes
from bioflow_ai.models.tools import ToolDefine, ToolGroup, ToolImage, ToolMemory, ToolTag, ToolTagLink
from bioflow_ai.models.workflow import Workflow
from bioflow_ai.schemas.library import (
    LibraryStatsPublic,
    LibraryTaxonomyPublic,
    TaxonomyGroupItem,
    TaxonomyLinkItem,
    TaxonomyTagItem,
)
from bioflow_ai.schemas.workflow_config import WorkflowType


def _count(session: Session, column: object) -> int:
    return int(session.scalar(select(func.count(column))) or 0)


def get_library_stats(session: Session) -> LibraryStatsPublic:
    """Return aggregate counts across all node-library asset tables."""
    tools = _count(session, ToolDefine.uid)
    tool_groups = _count(session, ToolGroup.id)
    images = _count(session, ToolImage.uid)
    codes = _count(session, CodeDefine.uid)
    subgraphs = int(
        session.scalar(select(func.count(Workflow.uid)).where(Workflow.wf_type == WorkflowType.SUBMODULE)) or 0
    )
    tool_memories = _count(session, ToolMemory.uid)
    bio_dbs = _count(session, BioDb.id)
    reference_genomes = _count(session, ReferenceGenomes.id)

    return LibraryStatsPublic(
        tools=tools,
        tool_groups=tool_groups,
        images=images,
        codes=codes,
        subgraphs=subgraphs,
        executable_nodes=tools + codes + subgraphs,
        tool_memories=tool_memories,
        bio_dbs=bio_dbs,
        reference_genomes=reference_genomes,
        resources=bio_dbs + reference_genomes,
    )


def get_library_taxonomy(session: Session, max_tags: int = 36) -> LibraryTaxonomyPublic:
    """Return tool groups, top tags, and group-tag co-occurrence links."""
    group_count_col = func.count(ToolDefine.uid).label('tool_count')
    group_rows = session.execute(
        select(
            ToolGroup.id,
            ToolGroup.name,
            ToolGroup.parent_id,
            group_count_col,
        )
        .outerjoin(ToolDefine, ToolGroup.id == ToolDefine.group_id)
        .group_by(ToolGroup.id, ToolGroup.name, ToolGroup.parent_id)
        .order_by(group_count_col.desc(), ToolGroup.name.asc())
    ).mappings()

    groups = [
        TaxonomyGroupItem(
            id=int(row['id']),
            name=str(row['name']),
            parent_id=int(row['parent_id']) if row['parent_id'] is not None else None,
            tool_count=int(row['tool_count']),
        )
        for row in group_rows
    ]

    tag_count_col = func.count(ToolTagLink.tool_uid).label('tool_count')
    tag_rows = list(
        session.execute(
            select(
                ToolTag.id,
                ToolTag.name,
                tag_count_col,
            )
            .join(ToolTagLink, ToolTag.id == ToolTagLink.tag_id)
            .group_by(ToolTag.id, ToolTag.name)
            .order_by(tag_count_col.desc(), ToolTag.name.asc())
            .limit(max_tags)
        ).mappings()
    )

    tags = [
        TaxonomyTagItem(
            id=int(row['id']),
            name=str(row['name']),
            tool_count=int(row['tool_count']),
        )
        for row in tag_rows
    ]

    top_tag_ids = [tag.id for tag in tags]
    links: list[TaxonomyLinkItem] = []
    if top_tag_ids:
        link_weight_col = func.count(ToolDefine.uid).label('weight')
        link_rows = session.execute(
            select(
                ToolDefine.group_id,
                ToolTagLink.tag_id,
                link_weight_col,
            )
            .join(ToolTagLink, ToolDefine.uid == ToolTagLink.tool_uid)
            .where(ToolDefine.group_id.is_not(None), ToolTagLink.tag_id.in_(top_tag_ids))
            .group_by(ToolDefine.group_id, ToolTagLink.tag_id)
            .order_by(link_weight_col.desc())
        ).mappings()
        links = [
            TaxonomyLinkItem(
                group_id=int(row['group_id']),
                tag_id=int(row['tag_id']),
                weight=int(row['weight']),
            )
            for row in link_rows
            if row['group_id'] is not None
        ]

    return LibraryTaxonomyPublic(groups=groups, tags=tags, links=links)
