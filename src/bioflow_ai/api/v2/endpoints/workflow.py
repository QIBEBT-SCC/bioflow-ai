"""Version 2 workflow execution, status, output, and monitoring endpoints."""

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from anyio import Path as AsyncPath
from fastapi import APIRouter, HTTPException, Query
from fastapi import Path as PathParam
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.requests import Request
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_409_CONFLICT,
    HTTP_422_UNPROCESSABLE_CONTENT,
)

from bioflow_ai.core.auth import MemberDep
from bioflow_ai.crud import node_run as node_run_crud
from bioflow_ai.crud import project_workflow as project_workflow_crud
from bioflow_ai.crud import workflow_run as run_crud
from bioflow_ai.crud.resource import sample as sample_crud
from bioflow_ai.infrastructure.database import SessionDep
from bioflow_ai.infrastructure.redis.client import monitor_redis_client
from bioflow_ai.models.project import Project
from bioflow_ai.models.workflow import NodeRun, Workflow, WorkflowRun
from bioflow_ai.schemas.project_workflow import WorkflowRunResult
from bioflow_ai.schemas.react_flow import WorkflowDefinition
from bioflow_ai.schemas.user import CurrentUser
from bioflow_ai.schemas.user_config import UserRole
from bioflow_ai.schemas.workflow_config import ExecutionScope, ToolOutput
from bioflow_ai.schemas.workflow_run import (
    TERMINAL_NODE_STATUSES,
    FileContentRequest,
    FilePreviewMetaRequest,
    FilePreviewRegionRequest,
    FileTablePreviewMetaRequest,
    FileTablePreviewProfileRequest,
    FileTablePreviewQueryRequest,
    FileVcfPreviewDetailRequest,
    FileVcfPreviewMetaRequest,
    FileVcfPreviewQueryRequest,
    MonitorPublic,
    NodeRunPublic,
    NodeRunRecord,
    NodeRunStatus,
    OutputFileNode,
    PaginatedNodeRuns,
    PaginatedWorkflowRuns,
    ProjectWorkflowRunRequest,
    RunStatus,
    WorkflowRunPublic,
    WorkflowRunStatistics,
)
from bioflow_ai.services.file_preview.annotation import (
    ANNOTATION_KINDS,
    AnnotationPreviewError,
    annotation_metadata,
    annotation_region,
)
from bioflow_ai.services.file_preview.browser import build_file_tree, serve_file
from bioflow_ai.services.file_preview.service import run_preview
from bioflow_ai.services.file_preview.signal import (
    TEXT_SIGNAL_KINDS,
    SignalPreviewError,
    signal_metadata,
    signal_region,
)
from bioflow_ai.services.file_preview.table import (
    TablePreviewError,
    profile_column,
    query_table,
    table_file_kind,
    table_metadata,
)
from bioflow_ai.services.file_preview.vcf import (
    VcfPreviewError,
    query_vcf,
    vcf_file_kind,
    vcf_metadata,
    vcf_record_detail,
)
from bioflow_ai.services.file_preview.worker import MAX_BED_BYTES, PreviewError, file_kind
from bioflow_ai.services.workflow import runs
from bioflow_ai.services.workflow.compiler import WorkflowCompileError
from bioflow_ai.services.workflow.runs import file_still_written
from bioflow_ai.services.workflow.runtime import node_log_path
from bioflow_ai.utils.file.paths import to_safe_path_segment

if TYPE_CHECKING:
    from bioflow_ai.models.resource import Sample

workflow_router = APIRouter()
project_router = APIRouter()
run_router = APIRouter()
node_router = APIRouter()


def _is_admin(user: CurrentUser) -> bool:
    return user.role == UserRole.ADMIN


def _owner_filter(user: CurrentUser) -> int | None:
    """Return the owner ID to filter listings by; admins see every owner."""
    return None if _is_admin(user) else user.id


def _owned_run(session: Session, run_uid: uuid.UUID, user: CurrentUser) -> WorkflowRun:
    run = run_crud.get(session, run_uid)
    if run is None or (run.owner_id != user.id and not _is_admin(user)):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow run not found')
    return run


def _owned_node(session: Session, node_uid: uuid.UUID, user: CurrentUser) -> NodeRun:
    node = node_run_crud.get(session, node_uid)
    if node is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow node run not found')
    _owned_run(session, node.run_uid, user)
    return node


def _owned_project(session: Session, project_id: int, user: CurrentUser) -> Project:
    project = session.get(Project, project_id)
    if project is None or (project.owner_id != user.id and not _is_admin(user)):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Project not found')
    return project


def _project_workflow(session: Session, project_id: int, workflow_uid: uuid.UUID) -> Workflow:
    project_workflow = project_workflow_crud.get(session, project_id, workflow_uid)
    if project_workflow is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow not found in this project')
    return project_workflow.workflow


@workflow_router.post('/run')
def run_editor_workflow(
    workflow: WorkflowDefinition,
    session: SessionDep,
    current_user: MemberDep,
    template_name: str = 'unname',
) -> WorkflowRunPublic:
    """Start an editor run without rewriting graph node IDs."""
    try:
        run = runs.create_editor_run(session, workflow, template_name, current_user)
    except WorkflowCompileError as error:
        raise HTTPException(HTTP_400_BAD_REQUEST, str(error)) from error
    return runs.to_public(session, run)


@project_router.post('/{project_id}/workflows/{workflow_uid}/run')
def run_project_workflow(
    project_id: int,
    workflow_uid: uuid.UUID,
    body: ProjectWorkflowRunRequest,
    session: SessionDep,
    current_user: MemberDep,
) -> WorkflowRunResult:
    """Create or rerun the current generation for a project workflow."""
    project = _owned_project(session, project_id, current_user)
    workflow = _project_workflow(session, project_id, workflow_uid)

    if workflow.execution_scope == ExecutionScope.PROJECT_LEVEL:
        if body.sample_uids:
            raise HTTPException(HTTP_400_BAD_REQUEST, 'Project-level workflows do not accept sample_uids')
        selected_samples: list[Sample | None] = [None]
    else:
        if not body.sample_uids:
            raise HTTPException(HTTP_400_BAD_REQUEST, 'Sample-level workflows require sample_uids')
        selected_samples = list(sample_crud.get_by_uids(session, body.sample_uids, project_id=project_id))
        if len(selected_samples) != len(body.sample_uids):
            raise HTTPException(HTTP_404_NOT_FOUND, 'Some samples were not found in this project')

    try:
        return runs.create_project_runs(
            session,
            project_id=project_id,
            project_name=project.name,
            workflow=workflow,
            samples=selected_samples,
            current_user=current_user,
            run_name_prefix=body.run_name_prefix,
        )
    except runs.WorkflowRunConflictError as error:
        raise HTTPException(HTTP_409_CONFLICT, str(error)) from error
    except WorkflowCompileError as error:
        raise HTTPException(HTTP_400_BAD_REQUEST, str(error)) from error


@project_router.get('/{project_id}/workflows/{workflow_uid}/package')
def package_project_workflow_runs(
    project_id: int,
    workflow_uid: uuid.UUID,
    session: SessionDep,
    current_user: MemberDep,
) -> StreamingResponse:
    """Download only the current generation for each project/sample run."""
    _owned_project(session, project_id, current_user)
    workflow = _project_workflow(session, project_id, workflow_uid)
    owner_id = _owner_filter(current_user)
    try:
        content = runs.package_project_workflow_runs(
            session,
            project_id=project_id,
            workflow_uid=workflow_uid,
            owner_id=owner_id,
        )
    except runs.WorkflowRunNotFoundError as error:
        raise HTTPException(HTTP_404_NOT_FOUND, str(error)) from error
    return StreamingResponse(
        iter([content]),
        media_type='application/zip',
        headers={'Content-Disposition': f'attachment; filename="{to_safe_path_segment(workflow.name)}_results.zip"'},
    )


@run_router.get('')
def list_all_runs(
    session: SessionDep,
    current_user: MemberDep,
    offset: int = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    status: RunStatus | None = None,
) -> PaginatedWorkflowRuns:
    """List workflow runs."""
    owner_id = _owner_filter(current_user)
    return runs.list_runs(session, owner_id=owner_id, status=status, offset=offset, limit=limit)


@run_router.get('/statistics')
def get_run_statistics(session: SessionDep, current_user: MemberDep) -> WorkflowRunStatistics:
    """Get aggregate workflow run states."""
    owner_id = _owner_filter(current_user)
    return runs.run_statistics(session, owner_id=owner_id)


@project_router.get('/{project_id}/runs')
def list_project_runs(
    project_id: int,
    session: SessionDep,
    current_user: MemberDep,
    offset: int = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> PaginatedWorkflowRuns:
    """List current workflow generations for one owned project."""
    _owned_project(session, project_id, current_user)
    owner_id = _owner_filter(current_user)
    return runs.list_runs(session, owner_id=owner_id, project_id=project_id, offset=offset, limit=limit)


@project_router.get('/{project_id}/runs/stats')
def get_project_run_statistics(
    project_id: int,
    session: SessionDep,
    current_user: MemberDep,
) -> WorkflowRunStatistics:
    """Get aggregate workflow run states for an owned project."""
    _owned_project(session, project_id, current_user)
    owner_id = _owner_filter(current_user)
    return runs.run_statistics(session, owner_id=owner_id, project_id=project_id)


@run_router.get('/{uid}/node-runs')
def list_run_node_runs(
    uid: uuid.UUID,
    session: SessionDep,
    current_user: MemberDep,
) -> list[NodeRunRecord]:
    """List current-generation node executions for one workflow run."""
    return runs.list_node_runs_for_run(session, _owned_run(session, uid, current_user))


@run_router.get('/{uid}/foreach/{node_uid}/items')
def list_foreach_items(
    uid: uuid.UUID,
    node_uid: uuid.UUID,
    session: SessionDep,
    current_user: MemberDep,
    status: RunStatus | None = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict:
    """Page through one parent's child runs without embedding their graphs."""
    parent = _owned_run(session, uid, current_user)
    node = session.get(NodeRun, node_uid)
    if node is None or node.run_uid != parent.uid or node.node_type != 'foreach':
        raise HTTPException(HTTP_404_NOT_FOUND, 'ForEach node not found')
    query = select(WorkflowRun).where(WorkflowRun.parent_node_uid == node_uid)
    if status is not None:
        query = query.where(WorkflowRun.status == status)
    total = session.scalar(select(func.count()).select_from(query.subquery())) or 0
    children = session.scalars(query.order_by(WorkflowRun.item_index).offset(offset).limit(limit)).all()
    return {
        'total': total,
        'offset': offset,
        'limit': limit,
        'has_more': offset + limit < total,
        'data': [
            {
                'key': child.item_key,
                'name': child.item_name,
                'run_uid': str(child.uid),
                'status': child.status.value,
                'started': child.start_time is not None,
                'settled': child.settled,
            }
            for child in children
        ],
    }


@run_router.get('/{uid}')
def get_run(uid: uuid.UUID, session: SessionDep, current_user: MemberDep) -> WorkflowRunPublic:
    """Get the current generation and its injected graph state."""
    return runs.to_public(session, _owned_run(session, uid, current_user))


async def _stream_run_updates(
    uid: uuid.UUID,
    session: Session,
    user: CurrentUser,
    request: Request,
) -> AsyncGenerator[str]:
    while not await request.is_disconnected():
        session.expire_all()
        run = _owned_run(session, uid, user)
        public = runs.to_public(session, run)
        yield f'data: {public.model_dump_json()}\n\n'
        if public.settled:
            return
        await asyncio.sleep(2)


@run_router.get('/{uid}/stream')
def stream_run(
    uid: uuid.UUID,
    request: Request,
    session: SessionDep,
    current_user: MemberDep,
) -> StreamingResponse:
    """Stream through FAILED/unsettled and close only after settled=true."""
    _owned_run(session, uid, current_user)
    return StreamingResponse(
        _stream_run_updates(uid, session, current_user, request),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@run_router.get('/{uid}/files')
async def get_run_files(
    uid: uuid.UUID,
    session: SessionDep,
    current_user: MemberDep,
) -> list[OutputFileNode]:
    """List current-generation output files."""
    run = _owned_run(session, uid, current_user)
    base = AsyncPath(run.base_dir)
    if not await base.exists():
        return []
    return await build_file_tree(base, base)


@run_router.post('/{uid}/files/content')
async def get_run_file_content(
    uid: uuid.UUID,
    body: FileContentRequest,
    session: SessionDep,
    current_user: MemberDep,
) -> Response:
    """Serve a current-generation file without allowing path traversal."""
    run = _owned_run(session, uid, current_user)
    full_path = await _run_output_file(run, body.path)
    return await serve_file(full_path)


async def _run_output_file(run: WorkflowRun, path: str) -> AsyncPath:
    """Resolve one output path inside its run, including symlink targets."""
    base = await AsyncPath(run.base_dir).resolve()
    full_path = await (base / path).resolve()
    if not full_path.is_relative_to(base):
        raise HTTPException(HTTP_403_FORBIDDEN, 'Access denied')
    if not await full_path.is_file():
        raise HTTPException(HTTP_404_NOT_FOUND, 'File not found')
    return full_path


@run_router.get('/{uid}/files/view/{generation}/{path:path}')
async def view_run_file(
    uid: uuid.UUID,
    generation: int,
    path: str,
    session: SessionDep,
    current_user: MemberDep,
) -> Response:
    """Serve an owned output at a stable URL so HTML-relative assets resolve."""
    run = _owned_run(session, uid, current_user)
    if generation != run.generation:
        raise HTTPException(HTTP_409_CONFLICT, 'Run was restarted. Refresh its file tree.')
    full_path = await _run_output_file(run, path)
    response = await serve_file(full_path, inline=True)
    response.headers['Cache-Control'] = 'private, no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    suffix = full_path.suffix.lower()
    if suffix in {'.html', '.htm'}:
        response.headers['Content-Security-Policy'] = 'sandbox allow-scripts allow-same-origin'
    elif suffix == '.svg':
        response.headers['Content-Security-Policy'] = 'sandbox'
    return response


async def _preview_file(
    run: WorkflowRun,
    path: str,
    generation: int,
    session: Session,
) -> tuple[AsyncPath, AsyncPath, str]:
    """Apply run ownership's current-generation and path restrictions before parsing."""
    if generation != run.generation:
        raise HTTPException(HTTP_409_CONFLICT, 'Run was restarted. Refresh its file tree.')
    base = await AsyncPath(run.base_dir).resolve()
    full_path = await _run_output_file(run, path)
    try:
        kind = file_kind(Path(str(full_path)))
    except PreviewError:
        kind = 'unsupported'
    nodes = session.execute(select(NodeRun).where(NodeRun.run_uid == run.uid)).scalars()
    matching = [
        node for node in nodes if node.base_dir and full_path.is_relative_to(await AsyncPath(node.base_dir).resolve())
    ]
    if file_still_written(matching):
        raise HTTPException(HTTP_409_CONFLICT, 'This file belongs to a node that has not completed successfully.')
    if not matching and not run.settled:
        raise HTTPException(HTTP_409_CONFLICT, 'This file may still be written. Wait for the run to finish.')
    return base, full_path, kind


@run_router.post('/{uid}/files/preview/meta')
async def get_run_file_preview_meta(
    uid: uuid.UUID,
    body: FilePreviewMetaRequest,
    session: SessionDep,
    current_user: MemberDep,
) -> dict:
    """Report bounded, searchable genomic preview metadata."""
    run = _owned_run(session, uid, current_user)
    base, path, kind = await _preview_file(run, body.path, body.generation, session)
    if kind == 'unsupported':
        return {
            'kind': kind,
            'supported': False,
            'reason': 'This file format has no genomic preview.',
            'chromosomes': [],
            'default_chromosome': None,
            'has_more_chromosomes': False,
        }
    if kind == 'bed' and (await path.stat()).st_size > MAX_BED_BYTES:
        return {
            'kind': kind,
            'supported': False,
            'reason': 'BED preview supports files up to 4 MiB.',
            'chromosomes': [],
            'default_chromosome': None,
            'has_more_chromosomes': False,
        }
    if kind in TEXT_SIGNAL_KINDS:
        try:
            return await asyncio.to_thread(signal_metadata, Path(str(path)), kind, body.query)
        except SignalPreviewError as error:
            return {
                'kind': kind,
                'supported': False,
                'reason': str(error),
                'chromosomes': [],
                'default_chromosome': None,
                'has_more_chromosomes': False,
            }
    if kind in ANNOTATION_KINDS:
        try:
            return await asyncio.to_thread(annotation_metadata, Path(str(path)), kind, body.query)
        except AnnotationPreviewError as error:
            return {
                'kind': kind,
                'supported': False,
                'reason': str(error),
                'chromosomes': [],
                'default_chromosome': None,
                'has_more_chromosomes': False,
            }
    try:
        return await run_preview(base, path, 'meta', query=body.query)
    except HTTPException as error:
        if error.status_code == HTTP_422_UNPROCESSABLE_CONTENT:
            return {
                'kind': kind,
                'supported': False,
                'reason': error.detail,
                'chromosomes': [],
                'default_chromosome': None,
                'has_more_chromosomes': False,
            }
        raise


@run_router.post('/{uid}/files/preview/region')
async def get_run_file_preview_region(
    uid: uuid.UUID,
    body: FilePreviewRegionRequest,
    session: SessionDep,
    current_user: MemberDep,
) -> dict:
    """Read only one capped genomic viewport."""
    run = _owned_run(session, uid, current_user)
    base, path, kind = await _preview_file(run, body.path, body.generation, session)
    if kind == 'unsupported':
        raise HTTPException(415, 'This file format has no genomic preview.')
    if kind == 'bed' and (await path.stat()).st_size > MAX_BED_BYTES:
        raise HTTPException(413, 'BED preview supports files up to 4 MiB.')
    if body.end <= body.start:
        raise HTTPException(422, 'Use a non-empty, 0-based half-open interval.')
    max_span = 10_000_000 if kind in {'bigwig', 'bedgraph', 'wig', 'hic', 'cool', 'mcool'} else 1_000_000
    if body.end - body.start > max_span:
        raise HTTPException(413, f'This track allows a region of at most {max_span:,} bases.')
    if kind in {'hic', 'cool', 'mcool'}:
        if (body.start2 is None) != (body.end2 is None):
            raise HTTPException(422, 'Both coordinates of the second locus are required.')
        if body.start2 and body.end2 and (body.end2 <= body.start2 or body.end2 - body.start2 > max_span):
            raise HTTPException(413, 'The second locus must span 1 base to 10 Mb.')
    if kind in TEXT_SIGNAL_KINDS:
        try:
            return await asyncio.to_thread(
                signal_region,
                Path(str(path)),
                kind,
                body.chrom,
                body.start,
                body.end,
                body.bins,
            )
        except SignalPreviewError as error:
            raise HTTPException(422, str(error)) from error
    if kind in ANNOTATION_KINDS:
        try:
            return await asyncio.to_thread(
                annotation_region,
                Path(str(path)),
                kind,
                body.chrom,
                body.start,
                body.end,
            )
        except AnnotationPreviewError as error:
            raise HTTPException(422, str(error)) from error
    return await run_preview(
        base,
        path,
        'region',
        chrom=body.chrom,
        start=body.start,
        end=body.end,
        bins=body.bins,
        chrom2=body.chrom2,
        start2=body.start2,
        end2=body.end2,
        resolution=body.resolution,
    )


def _require_table_file(path: AsyncPath) -> None:
    """Reject non-tabular extensions before loading the file."""
    try:
        table_file_kind(Path(str(path)))
    except TablePreviewError as error:
        raise HTTPException(415, str(error)) from error


@run_router.post('/{uid}/files/preview/table/meta')
async def get_run_table_preview_meta(
    uid: uuid.UUID,
    body: FileTablePreviewMetaRequest,
    session: SessionDep,
    current_user: MemberDep,
) -> dict:
    """Return CSV/TSV schema metadata inferred by Polars."""
    run = _owned_run(session, uid, current_user)
    _, path, _ = await _preview_file(run, body.path, body.generation, session)
    _require_table_file(path)
    try:
        return await asyncio.to_thread(table_metadata, Path(str(path)))
    except TablePreviewError as error:
        raise HTTPException(422, str(error)) from error


@run_router.post('/{uid}/files/preview/table/query')
async def query_run_table_preview(
    uid: uuid.UUID,
    body: FileTablePreviewQueryRequest,
    session: SessionDep,
    current_user: MemberDep,
) -> dict:
    """Return one projected page after server-side filtering and sorting."""
    run = _owned_run(session, uid, current_user)
    _, path, _ = await _preview_file(run, body.path, body.generation, session)
    _require_table_file(path)
    try:
        return await asyncio.to_thread(
            query_table,
            Path(str(path)),
            offset=body.offset,
            limit=body.limit,
            columns=body.columns,
            sort=body.sort.model_dump() if body.sort else None,
            filters=[item.model_dump() for item in body.filters],
        )
    except TablePreviewError as error:
        raise HTTPException(422, str(error)) from error


@run_router.post('/{uid}/files/preview/table/profile')
async def profile_run_table_column(
    uid: uuid.UUID,
    body: FileTablePreviewProfileRequest,
    session: SessionDep,
    current_user: MemberDep,
) -> dict:
    """Compute an on-demand summary of one CSV/TSV column."""
    run = _owned_run(session, uid, current_user)
    _, path, _ = await _preview_file(run, body.path, body.generation, session)
    _require_table_file(path)
    try:
        return await asyncio.to_thread(
            profile_column,
            Path(str(path)),
            column=body.column,
            filters=[item.model_dump() for item in body.filters],
        )
    except TablePreviewError as error:
        raise HTTPException(422, str(error)) from error


def _require_vcf_file(path: AsyncPath) -> None:
    """Reject files without a VCF or compressed-VCF extension."""
    try:
        vcf_file_kind(Path(str(path)))
    except VcfPreviewError as error:
        raise HTTPException(415, str(error)) from error


@run_router.post('/{uid}/files/preview/vcf/meta')
async def get_run_vcf_preview_meta(
    uid: uuid.UUID,
    body: FileVcfPreviewMetaRequest,
    session: SessionDep,
    current_user: MemberDep,
) -> dict:
    """Return VCF header, variant summaries, and contig density data."""
    run = _owned_run(session, uid, current_user)
    _, path, _ = await _preview_file(run, body.path, body.generation, session)
    _require_vcf_file(path)
    try:
        return await asyncio.to_thread(vcf_metadata, Path(str(path)))
    except VcfPreviewError as error:
        raise HTTPException(422, str(error)) from error


@run_router.post('/{uid}/files/preview/vcf/query')
async def query_run_vcf_preview(
    uid: uuid.UUID,
    body: FileVcfPreviewQueryRequest,
    session: SessionDep,
    current_user: MemberDep,
) -> dict:
    """Return one filtered page of VCF records."""
    run = _owned_run(session, uid, current_user)
    _, path, _ = await _preview_file(run, body.path, body.generation, session)
    _require_vcf_file(path)
    try:
        return await asyncio.to_thread(
            query_vcf,
            Path(str(path)),
            offset=body.offset,
            limit=body.limit,
            contig=body.contig,
            variant_type=body.variant_type,
            filter_status=body.filter_status,
            search=body.search,
            min_qual=body.min_qual,
        )
    except VcfPreviewError as error:
        raise HTTPException(422, str(error)) from error


@run_router.post('/{uid}/files/preview/vcf/detail')
async def get_run_vcf_preview_detail(
    uid: uuid.UUID,
    body: FileVcfPreviewDetailRequest,
    session: SessionDep,
    current_user: MemberDep,
) -> dict:
    """Return INFO and sample FORMAT values for one VCF record."""
    run = _owned_run(session, uid, current_user)
    _, path, _ = await _preview_file(run, body.path, body.generation, session)
    _require_vcf_file(path)
    try:
        return await asyncio.to_thread(vcf_record_detail, Path(str(path)), body.record_index)
    except VcfPreviewError as error:
        raise HTTPException(422, str(error)) from error


@node_router.get('')
def list_node_runs(
    session: SessionDep,
    current_user: MemberDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    status: NodeRunStatus | None = None,
) -> PaginatedNodeRuns:
    """List current-generation node executions for the task center."""
    owner_id = _owner_filter(current_user)
    return runs.list_node_runs(session, owner_id=owner_id, status=status, offset=offset, limit=limit)


@node_router.get('/recent/{hours}')
def list_recent_node_runs(
    hours: Annotated[int, PathParam(ge=1, le=24 * 7)],
    session: SessionDep,
    current_user: MemberDep,
    limit: Annotated[int, Query(ge=1, le=1000)] = 1000,
) -> list[NodeRunRecord]:
    """List recent current-generation node executions for the task timeline."""
    owner_id = _owner_filter(current_user)
    return runs.recent_node_runs(session, owner_id=owner_id, hours=hours, limit=limit)


@node_router.get('/{uid}')
def get_node_run(uid: uuid.UUID, session: SessionDep, current_user: MemberDep) -> NodeRunPublic:
    """Get one current-generation node run."""
    return runs.to_node_public(session, _owned_node(session, uid, current_user))


@node_router.get('/{uid}/result')
def get_node_result(uid: uuid.UUID, session: SessionDep, current_user: MemberDep) -> ToolOutput:
    """Get a completed node output from PostgreSQL."""
    node = _owned_node(session, uid, current_user)
    if node.tool_output is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Node has no output')
    return node.tool_output


@node_router.get('/{uid}/monitor')
async def get_node_monitor(
    uid: uuid.UUID,
    session: SessionDep,
    current_user: MemberDep,
    max_points: Annotated[int, Query(ge=1, le=1000)] = 300,
) -> list[MonitorPublic]:
    """Read transient resource samples; Redis is not used for DAG correctness."""
    _owned_node(session, uid, current_user)
    raw_list = await monitor_redis_client.lrange(f'node-monitor-{uid}', 0, -1)
    records = [MonitorPublic.model_validate(json.loads(record)) for record in raw_list]
    if len(records) > max_points:
        records = records[:: max(1, len(records) // max_points)]
    return records


@node_router.get('/{uid}/log')
def get_node_log(uid: uuid.UUID, session: SessionDep, current_user: MemberDep) -> dict[str, str | int]:
    """Read the durable node log if one exists."""
    path = node_log_path(_owned_node(session, uid, current_user))
    if path is None or not path.exists():
        return {'content': '', 'offset': 0}
    with path.open('r', encoding='utf-8') as log_file:
        content = log_file.read()
        offset = log_file.tell()
    return {'content': content, 'offset': offset}


async def _stream_node_log(
    uid: uuid.UUID,
    session: Session,
    user: CurrentUser,
    request: Request,
    offset: int,
) -> AsyncGenerator[str]:
    while not await request.is_disconnected():
        session.expire_all()
        node = _owned_node(session, uid, user)
        path = node_log_path(node)
        content = ''
        if path is not None and path.exists():
            with path.open('r', encoding='utf-8') as log_file:
                log_file.seek(offset)
                content = log_file.read()
                offset = log_file.tell()
        if content:
            yield f'data: {json.dumps({"content": content})}\n\n'
        if node.status in TERMINAL_NODE_STATUSES:
            return
        await asyncio.sleep(2)


@node_router.get('/{uid}/log/stream')
def stream_node_log(
    uid: uuid.UUID,
    request: Request,
    session: SessionDep,
    current_user: MemberDep,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> StreamingResponse:
    """Stream log additions until the node reaches a terminal state."""
    _owned_node(session, uid, current_user)
    return StreamingResponse(
        _stream_node_log(uid, session, current_user, request, offset),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )
