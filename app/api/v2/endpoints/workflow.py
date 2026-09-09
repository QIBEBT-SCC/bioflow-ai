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
from sqlalchemy.orm import Session
from starlette.requests import Request
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_403_FORBIDDEN, HTTP_404_NOT_FOUND, HTTP_409_CONFLICT

from app.core.auth import MemberDep
from app.core.redis_client import monitor_redis_client
from app.core.session import SessionDep
from app.crud import project_workflow as project_workflow_crud
from app.crud import workflow as workflow_crud
from app.crud.resource import sample as sample_crud
from app.models.project import Project
from app.models.workflow import NodeRun, WorkflowRun
from app.schemas.project_workflow import WorkflowRunResult
from app.schemas.react_flow import WorkflowDefinition
from app.schemas.user import CurrentUser
from app.schemas.user_config import UserRole
from app.schemas.workflow_config import ExecutionScope, ToolOutput
from app.schemas.workflow_run import (
    FileContentRequest,
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
from app.services.workflow import runs
from app.services.workflow.compiler import WorkflowCompileError
from app.utils.file_browser import build_file_tree, serve_file
from app.utils.input_format import to_safe_path_segment

if TYPE_CHECKING:
    from app.models.resource import Sample

workflow_router = APIRouter()
project_router = APIRouter()
run_router = APIRouter()
node_router = APIRouter()


def _is_admin(user: CurrentUser) -> bool:
    return user.role == UserRole.ADMIN


def _owned_run(session: Session, run_uid: uuid.UUID, user: CurrentUser) -> WorkflowRun:
    run = runs.get_run(session, run_uid)
    if run is None or (run.owner_id != user.id and not _is_admin(user)):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow run not found')
    return run


def _owned_node(session: Session, node_uid: uuid.UUID, user: CurrentUser) -> NodeRun:
    node = runs.get_node(session, node_uid)
    if node is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow node run not found')
    _owned_run(session, node.run_uid, user)
    return node


def _owned_project(session: Session, project_id: int, user: CurrentUser) -> Project:
    project = session.get(Project, project_id)
    if project is None or (project.owner_id != user.id and not _is_admin(user)):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Project not found')
    return project


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
    if not project_workflow_crud.exists(session, project_id, workflow_uid):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow not found in this project')
    workflow = workflow_crud.get(session, workflow_uid)
    if workflow is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow not found')

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
    if not project_workflow_crud.exists(session, project_id, workflow_uid):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow not found in this project')
    workflow = workflow_crud.get(session, workflow_uid)
    if workflow is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Workflow not found')
    owner_id = None if _is_admin(current_user) else current_user.id
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
    owner_id = None if _is_admin(current_user) else current_user.id
    return runs.list_runs(session, owner_id=owner_id, status=status, offset=offset, limit=limit)


@run_router.get('/statistics')
def get_run_statistics(session: SessionDep, current_user: MemberDep) -> WorkflowRunStatistics:
    """Get aggregate workflow run states."""
    owner_id = None if _is_admin(current_user) else current_user.id
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
    owner_id = None if _is_admin(current_user) else current_user.id
    return runs.list_runs(session, owner_id=owner_id, project_id=project_id, offset=offset, limit=limit)


@project_router.get('/{project_id}/runs/stats')
def get_project_run_statistics(
    project_id: int,
    session: SessionDep,
    current_user: MemberDep,
) -> WorkflowRunStatistics:
    """Get aggregate workflow run states for an owned project."""
    _owned_project(session, project_id, current_user)
    owner_id = None if _is_admin(current_user) else current_user.id
    return runs.run_statistics(session, owner_id=owner_id, project_id=project_id)


@run_router.get('/{uid}/node-runs')
def list_run_node_runs(
    uid: uuid.UUID,
    session: SessionDep,
    current_user: MemberDep,
) -> list[NodeRunRecord]:
    """List current-generation node executions for one workflow run."""
    return runs.list_node_runs_for_run(session, _owned_run(session, uid, current_user))


@run_router.get('/{uid}')
def get_run(uid: uuid.UUID, session: SessionDep, current_user: MemberDep) -> WorkflowRunPublic:
    """Get the current generation and its injected graph state."""
    return runs.to_public(session, _owned_run(session, uid, current_user))


async def _stream_run_updates(
    uid: uuid.UUID,
    session: Session,
    user: CurrentUser,
    request: Request,
) -> AsyncGenerator[str, None]:
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
    base = await AsyncPath(run.base_dir).resolve()
    full_path = await (base / body.path).resolve()
    if not full_path.is_relative_to(base):
        raise HTTPException(HTTP_403_FORBIDDEN, 'Access denied')
    if not await full_path.exists() or not await full_path.is_file():
        raise HTTPException(HTTP_404_NOT_FOUND, 'File not found')
    return await serve_file(full_path)


@node_router.get('')
def list_node_runs(
    session: SessionDep,
    current_user: MemberDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    status: NodeRunStatus | None = None,
) -> PaginatedNodeRuns:
    """List current-generation node executions for the task center."""
    owner_id = None if _is_admin(current_user) else current_user.id
    return runs.list_node_runs(session, owner_id=owner_id, status=status, offset=offset, limit=limit)


@node_router.get('/recent/{hours}')
def list_recent_node_runs(
    hours: Annotated[int, PathParam(ge=1, le=24 * 7)],
    session: SessionDep,
    current_user: MemberDep,
    limit: Annotated[int, Query(ge=1, le=1000)] = 1000,
) -> list[NodeRunRecord]:
    """List recent current-generation node executions for the task timeline."""
    owner_id = None if _is_admin(current_user) else current_user.id
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
    # pyrefly: ignore [not-async]
    raw_list = await monitor_redis_client.lrange(f'node-monitor-{uid}', 0, -1)
    records = [MonitorPublic.model_validate(json.loads(record)) for record in raw_list]
    if len(records) > max_points:
        records = records[:: max(1, len(records) // max_points)]
    return records


def _node_log_path(node: NodeRun) -> Path | None:
    if node.tool_output is not None and node.tool_output.log:
        return Path(node.tool_output.log)
    return Path(node.base_dir) / '11_run.log' if node.base_dir else None


@node_router.get('/{uid}/log')
def get_node_log(uid: uuid.UUID, session: SessionDep, current_user: MemberDep) -> dict[str, str | int]:
    """Read the durable node log if one exists."""
    path = _node_log_path(_owned_node(session, uid, current_user))
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
) -> AsyncGenerator[str, None]:
    while not await request.is_disconnected():
        session.expire_all()
        node = _owned_node(session, uid, user)
        path = _node_log_path(node)
        content = ''
        if path is not None and path.exists():
            with path.open('r', encoding='utf-8') as log_file:
                log_file.seek(offset)
                content = log_file.read()
                offset = log_file.tell()
        if content:
            yield f'data: {json.dumps({"content": content})}\n\n'
        if node.status in {NodeRunStatus.SUCCEEDED, NodeRunStatus.FAILED, NodeRunStatus.BLOCKED}:
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
