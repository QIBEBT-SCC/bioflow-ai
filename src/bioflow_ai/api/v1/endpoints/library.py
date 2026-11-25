from typing import Annotated

from fastapi import APIRouter, Query

from bioflow_ai.infrastructure.database import SessionDep
from bioflow_ai.schemas.library import LibraryStatsPublic, LibraryTaxonomyPublic
from bioflow_ai.services.library import get_library_stats, get_library_taxonomy

router = APIRouter()


@router.get('/stats')
def read_library_stats(session: SessionDep) -> LibraryStatsPublic:
    """Get aggregate counts of tools, nodes, images, memories, and resources."""
    return get_library_stats(session)


@router.get('/taxonomy')
def read_library_taxonomy(
    session: SessionDep,
    max_tags: Annotated[int, Query(ge=5, le=80)] = 36,
) -> LibraryTaxonomyPublic:
    """Get tool groups, top tags, and group-tag co-occurrence weights."""
    return get_library_taxonomy(session, max_tags=max_tags)
