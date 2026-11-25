from pydantic import BaseModel


class LibraryStatsPublic(BaseModel):
    """Aggregate node-library counts exposed for the platform overview."""

    tools: int
    tool_groups: int
    images: int
    codes: int
    subgraphs: int
    executable_nodes: int
    tool_memories: int
    bio_dbs: int
    reference_genomes: int
    resources: int


class TaxonomyGroupItem(BaseModel):
    """Tool group node with its curated tool count."""

    id: int
    name: str
    parent_id: int | None = None
    tool_count: int


class TaxonomyTagItem(BaseModel):
    """Tool tag satellite node with its associated tool count."""

    id: int
    name: str
    tool_count: int


class TaxonomyLinkItem(BaseModel):
    """Co-occurrence edge connecting a tool group and a tool tag."""

    group_id: int
    tag_id: int
    weight: int


class LibraryTaxonomyPublic(BaseModel):
    """Group-tag taxonomy graph data for the Node Library atlas."""

    groups: list[TaxonomyGroupItem]
    tags: list[TaxonomyTagItem]
    links: list[TaxonomyLinkItem]
