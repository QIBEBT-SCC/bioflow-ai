from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from starlette.status import HTTP_404_NOT_FOUND, HTTP_422_UNPROCESSABLE_CONTENT

import app.crud.resource.reference_genome as genome_crud
from app.core.redis_client import monitor_redis_client
from app.core.session import SessionDep
from app.models.resource import ReferenceGenomes
from app.schemas.resource import (
    CeleryTaskResponse,
    GenomeDownloadResponse,
    GenomeIndexStatus,
    IndexStatus,
    ReferenceGenomeBuildIndexRequest,
    ReferenceGenomeDownloadRequest,
    ReferenceGenomeListItem,
    ReferenceGenomePublic,
)
from app.tasks.load_tasks.load_genome import build_genome_index_admin_task, download_genome_admin_task
from app.utils.api.ncbi_datasets import get_species_info, search_reference_genomes, search_reference_genomes_by_tax_id
from app.utils.security import AdminDep, UserDep

reference_router = APIRouter()

# ── Redis key helpers ──────────────────────────────────────────────────────────
_INDEX_NAMES = ('bowtie2', 'bwa', 'hisat2', 'star', 'minimap2')
_INDEX_FIELD_MAP = {
    'bowtie2': 'bowtie2_index',
    'bwa': 'bwa_index',
    'hisat2': 'hisat2_index',
    'star': 'star_index',
    'minimap2': 'minimap2_index',
}


def _redis_index_key(genome_id: int, index_name: str) -> str:
    """Return the Redis key used to track an in-progress index build task."""
    return f'genome:{genome_id}:index:{index_name}:task_id'


async def _build_index_status(genome: ReferenceGenomes) -> GenomeIndexStatus:
    """Compute per-index status by checking DB fields and Redis task tracking."""
    statuses: dict[str, IndexStatus] = {}
    for name in _INDEX_NAMES:
        field = _INDEX_FIELD_MAP[name]
        if getattr(genome, field, None):
            statuses[name] = IndexStatus.ready
        else:
            # pyrefly: ignore [bad-argument-type]
            key = _redis_index_key(genome.id, name)
            task_id = await monitor_redis_client.get(key)
            if task_id:
                statuses[name] = IndexStatus.building
            else:
                statuses[name] = IndexStatus.not_built
    return GenomeIndexStatus(**statuses)


async def _to_list_item(genome: ReferenceGenomes) -> ReferenceGenomeListItem:
    item = ReferenceGenomeListItem.model_validate(genome)
    item.index_status = await _build_index_status(genome)
    return item


async def _to_public(genome: ReferenceGenomes) -> ReferenceGenomePublic:
    pub = ReferenceGenomePublic.model_validate(genome)
    pub.index_status = await _build_index_status(genome)
    return pub


# ── Endpoints ──────────────────────────────────────────────────────────────────


@reference_router.get('')
async def list_references(
    session: SessionDep,
    _current_user: UserDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 20,
) -> list[ReferenceGenomeListItem]:
    """Get a paginated list of all locally available reference genomes with index status."""
    genomes = genome_crud.get_all(session, limit, offset)
    return [await _to_list_item(g) for g in genomes]


@reference_router.get('/count')
async def get_reference_count(session: SessionDep, _current_user: UserDep) -> int:
    """Get the total number of reference genomes stored in the database."""
    return genome_crud.get_count(session)


@reference_router.get('/search')
async def search_references(
    session: SessionDep,
    _current_user: UserDep,
    q: str,
) -> list[ReferenceGenomeListItem]:
    """Search local reference genomes by species name or alias (fuzzy match) with index status."""
    genomes = genome_crud.search(session, q)
    return [await _to_list_item(g) for g in genomes]


@reference_router.get('/{ref_id}')
async def get_reference_by_id(ref_id: int, session: SessionDep, _current_user: UserDep) -> ReferenceGenomePublic:
    """Get full details and per-index build status of a reference genome by its database ID."""
    genome = genome_crud.get(session, ref_id)
    if not genome:
        raise HTTPException(HTTP_404_NOT_FOUND, f'Reference genome with id={ref_id} not found.')
    return await _to_public(genome)


@reference_router.delete('/{ref_id}')
async def delete_reference(ref_id: int, session: SessionDep, _current_user: AdminDep) -> dict[str, str]:
    """Delete a reference genome record from the database (files on disk are preserved)."""
    deleted = genome_crud.delete(session, ref_id)
    if not deleted:
        raise HTTPException(HTTP_404_NOT_FOUND, f'Reference genome with id={ref_id} not found.')
    return {'message': f'Reference genome id={ref_id} deleted successfully.'}


@reference_router.post('/download')
async def download_reference(
    request: ReferenceGenomeDownloadRequest,
    session: SessionDep,
    _current_user: AdminDep,
) -> GenomeDownloadResponse:
    """Trigger a background Celery task to download a new reference genome from NCBI.

    Performs a fast NCBI metadata search synchronously first to validate the target
    exists and to return genome info immediately. The actual download runs in the
    background. Supply exactly one of `ncbi_tax_id` or `species_name`.
    """
    if request.ncbi_tax_id is None and request.species_name is None:
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_CONTENT,
            "Either 'ncbi_tax_id' or 'species_name' must be provided.",
        )

    # --- Synchronous NCBI metadata lookup (fast, metadata only) ---------------
    try:
        if request.ncbi_tax_id is not None:
            genome_meta = search_reference_genomes_by_tax_id(request.ncbi_tax_id)
        else:
            # pyrefly: ignore [bad-argument-type]
            genome_meta = search_reference_genomes(request.species_name)
        species_info = get_species_info(genome_meta.organism.tax_id)
    except ValueError as e:
        raise HTTPException(HTTP_404_NOT_FOUND, str(e)) from e
    except Exception as e:
        raise HTTPException(HTTP_422_UNPROCESSABLE_CONTENT, f'NCBI lookup failed: {e}') from e

    # Check if already in DB
    existing = genome_crud.get_by_taxid(session, species_info.tax_id)
    if existing and not request.required_index:
        return GenomeDownloadResponse(
            task_id='',
            message=f'Genome "{existing.name}" already exists in the database (id={existing.id}). '
            'Submit required_index to build additional indexes.',
            ncbi_accession=existing.accession,
            ncbi_tax_id=existing.ncbi_tax_id,
            species_name=existing.name,
        )

    # --- Dispatch Celery task --------------------------------------------------
    task = download_genome_admin_task.apply_async(
        kwargs={
            'ncbi_tax_id': request.ncbi_tax_id,
            'species_name': request.species_name,
            'required_index': request.required_index,
        }
    )

    # Track building indexes in Redis (TTL: 24 h)
    if existing and request.required_index:
        for index_name in request.required_index:
            short = index_name.replace('_index', '')
            if short in _INDEX_NAMES:
                # pyrefly: ignore [bad-argument-type]
                key = _redis_index_key(existing.id, short)
                await monitor_redis_client.set(key, task.id, ex=86400)

    action = 'Download + index build' if request.required_index else 'Download'
    return GenomeDownloadResponse(
        task_id=task.id,
        message=f'{action} task submitted for "{species_info.organism_name}" '
        f'({genome_meta.accession}). Use task_id to check progress.',
        ncbi_accession=genome_meta.accession,
        ncbi_tax_id=species_info.tax_id,
        # pyrefly: ignore [bad-argument-type]
        species_name=species_info.organism_name,
    )


@reference_router.post('/{ref_id}/build-index')
async def build_reference_index(
    ref_id: int,
    request: ReferenceGenomeBuildIndexRequest,
    session: SessionDep,
    _current_user: AdminDep,
) -> CeleryTaskResponse:
    """Trigger a background Celery task to build alignment indexes for an existing reference genome."""
    genome = genome_crud.get(session, ref_id)
    if not genome:
        raise HTTPException(HTTP_404_NOT_FOUND, f'Reference genome with id={ref_id} not found.')

    task = build_genome_index_admin_task.apply_async(
        kwargs={
            'genome_id': ref_id,
            'required_index': request.required_index,
        }
    )

    # Track each requested build in Redis (TTL: 24 h)
    for index_name in request.required_index:
        short = index_name.replace('_index', '')
        if short in _INDEX_NAMES:
            key = _redis_index_key(ref_id, short)
            await monitor_redis_client.set(key, task.id, ex=86400)

    return CeleryTaskResponse(
        task_id=task.id,
        message=f'Index build task for "{genome.name}" submitted. Use task_id to check progress.',
    )
