import hashlib
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

from sqlalchemy.orm import Session

from bioflow_ai.core.config import get_settings
from bioflow_ai.crud import agent as agent_crud
from bioflow_ai.crud import project as project_crud
from bioflow_ai.crud import user as user_crud
from bioflow_ai.models.agent import AgentRun
from bioflow_ai.models.project import Project
from bioflow_ai.schemas.agent_artifact import AgentArtifactKind, AgentArtifactPublic
from bioflow_ai.schemas.user_config import UserRole

MAX_EDIT_BYTES: Final = 1024 * 1024
READ_ONLY_MESSAGE: Final = 'This agent file is read-only.'
ACTIVE_RUN_MESSAGE: Final = 'An active agent run is using the project plan.'
STALE_REVISION_MESSAGE: Final = 'The project plan changed after it was opened.'
EMPTY_PLAN_MESSAGE: Final = 'The project plan cannot be empty.'
PLAN_TOO_LARGE_MESSAGE: Final = 'The project plan exceeds the 1 MiB edit limit.'
PROJECT_DOCUMENTS: Final = {
    AgentArtifactKind.PLAN: 'PLAN.md',
    AgentArtifactKind.SAMPLES: 'SAMPLES.md',
}
RUN_ARTIFACT_FIELDS: Final = {
    AgentArtifactKind.DIAGNOSIS: ('workflow-assistant', 'diagnosis_path', 'DIAGNOSIS.md'),
    AgentArtifactKind.UPDATE: ('workflow-fixer', 'update_path', 'UPDATE.md'),
}


class AgentArtifactError(ValueError):
    """Base error for controlled agent file operations."""


class AgentArtifactNotFoundError(AgentArtifactError):
    """Raised when a file or its owning project cannot be accessed."""


class AgentArtifactReadOnlyError(AgentArtifactError):
    """Raised when a read-only agent file is edited."""


class AgentArtifactConflictError(AgentArtifactError):
    """Raised when an edit conflicts with an active run or stale revision."""


class AgentArtifactValidationError(AgentArtifactError):
    """Raised when proposed file content is invalid."""


def _is_admin_actor(session: Session, owner_id: int) -> bool:
    actor = user_crud.get_by_id(session, owner_id)
    return actor is not None and actor.role == UserRole.ADMIN


def get_project_root(project_id: int) -> Path:
    """Return the canonical persistent directory for a project."""
    storage_root = Path(get_settings().persistent.storage.STORAGE_PATH).expanduser().resolve()
    return storage_root / 'proj' / str(project_id)


def _get_owned_project(session: Session, project_id: int, owner_id: int) -> Project:
    project = project_crud.get_by_id(session, project_id)
    if project is None or (project.owner_id != owner_id and not _is_admin_actor(session, owner_id)):
        raise AgentArtifactNotFoundError
    return project


def _revision(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(64 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_regular_file(project_root: Path, candidate: Path) -> Path:
    root = project_root.resolve()
    unresolved = candidate if candidate.is_absolute() else root / candidate
    try:
        relative = unresolved.relative_to(root)
    except ValueError as error:
        raise AgentArtifactNotFoundError from error

    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise AgentArtifactNotFoundError

    resolved = unresolved.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise AgentArtifactNotFoundError
    return resolved


def _run_relative_path(run: AgentRun, kind: AgentArtifactKind) -> PurePosixPath:
    expected_agent, field_name, expected_name = RUN_ARTIFACT_FIELDS[kind]
    if run.agent_name != expected_agent:
        raise AgentArtifactNotFoundError
    raw_path = (run.result_payload or {}).get(field_name)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise AgentArtifactNotFoundError
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or '..' in relative.parts or relative.name != expected_name:
        raise AgentArtifactNotFoundError
    return relative


def _parse_artifact_id(artifact_id: str) -> tuple[AgentArtifactKind, uuid.UUID | None]:
    if artifact_id in {kind.value for kind in PROJECT_DOCUMENTS}:
        return AgentArtifactKind(artifact_id), None
    kind_value, separator, run_value = artifact_id.partition(':')
    if not separator or kind_value not in {kind.value for kind in RUN_ARTIFACT_FIELDS}:
        raise AgentArtifactNotFoundError
    try:
        return AgentArtifactKind(kind_value), uuid.UUID(run_value)
    except ValueError as error:
        raise AgentArtifactNotFoundError from error


def _resolve_artifact(
    session: Session,
    project_id: int,
    owner_id: int,
    artifact_id: str,
) -> tuple[Path, AgentArtifactKind, AgentRun | None]:
    _get_owned_project(session, project_id, owner_id)
    root = get_project_root(project_id)
    kind, run_uid = _parse_artifact_id(artifact_id)
    if kind in PROJECT_DOCUMENTS:
        return _safe_regular_file(root, root / PROJECT_DOCUMENTS[kind]), kind, None

    if run_uid is None:
        raise AgentArtifactNotFoundError
    run = agent_crud.get_run(session, run_uid)
    if (
        run is None
        or run.project_id != project_id
        or (run.agent_session.owner_id != owner_id and not _is_admin_actor(session, owner_id))
    ):
        raise AgentArtifactNotFoundError
    relative = _run_relative_path(run, kind)
    return _safe_regular_file(root, root / Path(*relative.parts)), kind, run


def _metadata(
    session: Session,
    project_id: int,
    path: Path,
    kind: AgentArtifactKind,
    run: AgentRun | None,
    *,
    active_run: bool | None = None,
) -> AgentArtifactPublic:
    if active_run is None:
        active_run = agent_crud.has_active_project_run(session, project_id)
    editable = kind == AgentArtifactKind.PLAN and not active_run
    if kind == AgentArtifactKind.PLAN:
        read_only_reason = 'active_agent_run' if active_run else None
    elif kind == AgentArtifactKind.SAMPLES:
        read_only_reason = 'managed_sample_summary'
    else:
        read_only_reason = 'immutable_audit_record'
    stat = path.stat()
    return AgentArtifactPublic(
        id=kind.value if run is None else f'{kind.value}:{run.uid}',
        kind=kind,
        name=path.name,
        project_id=project_id,
        run_uid=run.uid if run else None,
        agent_name=run.agent_name if run else None,
        run_status=run.status if run else None,
        size=stat.st_size,
        updated_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        content_type='text/markdown; charset=utf-8',
        revision=_revision(path),
        editable=editable,
        read_only_reason=read_only_reason,
    )


def list_project_artifacts(session: Session, project_id: int, owner_id: int) -> list[AgentArtifactPublic]:
    """List safe current project documents and immutable run reports."""
    _get_owned_project(session, project_id, owner_id)
    root = get_project_root(project_id)
    active_run = agent_crud.has_active_project_run(session, project_id)
    files: list[AgentArtifactPublic] = []
    for kind, name in PROJECT_DOCUMENTS.items():
        try:
            path = _safe_regular_file(root, root / name)
        except AgentArtifactNotFoundError:
            continue
        files.append(_metadata(session, project_id, path, kind, None, active_run=active_run))

    run_owner_id = None if _is_admin_actor(session, owner_id) else owner_id
    for run in agent_crud.list_project_artifact_runs(session, project_id, run_owner_id):
        kind = AgentArtifactKind.DIAGNOSIS if run.agent_name == 'workflow-assistant' else AgentArtifactKind.UPDATE
        try:
            relative = _run_relative_path(run, kind)
            path = _safe_regular_file(root, root / Path(*relative.parts))
        except AgentArtifactNotFoundError:
            continue
        files.append(_metadata(session, project_id, path, kind, run, active_run=active_run))
    return files


def list_run_artifacts(session: Session, run: AgentRun, owner_id: int) -> list[AgentArtifactPublic]:
    """List the current documents or reports relevant to one agent run."""
    if run.project_id is None:
        return []
    _get_owned_project(session, run.project_id, owner_id)
    if run.agent_session.owner_id != owner_id and not _is_admin_actor(session, owner_id):
        raise AgentArtifactNotFoundError
    root = get_project_root(run.project_id)
    if run.agent_name == 'project-planner':
        kinds = [AgentArtifactKind.PLAN, AgentArtifactKind.SAMPLES]
    elif run.agent_name == 'workflow-assistant':
        kinds = [AgentArtifactKind.DIAGNOSIS]
    elif run.agent_name == 'workflow-fixer':
        kinds = [AgentArtifactKind.UPDATE]
    else:
        return []

    active_run = agent_crud.has_active_project_run(session, run.project_id)
    files: list[AgentArtifactPublic] = []
    for kind in kinds:
        try:
            if kind in PROJECT_DOCUMENTS:
                path = _safe_regular_file(root, root / PROJECT_DOCUMENTS[kind])
                source_run = None
            else:
                relative = _run_relative_path(run, kind)
                path = _safe_regular_file(root, root / Path(*relative.parts))
                source_run = run
        except AgentArtifactNotFoundError:
            continue
        files.append(_metadata(session, run.project_id, path, kind, source_run, active_run=active_run))
    return files


def get_artifact(
    session: Session,
    project_id: int,
    owner_id: int,
    artifact_id: str,
) -> tuple[Path, AgentArtifactPublic]:
    """Resolve one owned logical file ID to a safe regular file."""
    path, kind, run = _resolve_artifact(session, project_id, owner_id, artifact_id)
    return path, _metadata(session, project_id, path, kind, run)


def update_artifact(
    session: Session,
    project_id: int,
    owner_id: int,
    artifact_id: str,
    content: str,
    expected_revision: str,
) -> AgentArtifactPublic:
    """Atomically update an editable agent file using optimistic concurrency."""
    _get_owned_project(session, project_id, owner_id)
    kind, run_uid = _parse_artifact_id(artifact_id)
    if kind != AgentArtifactKind.PLAN or run_uid is not None:
        raise AgentArtifactReadOnlyError(READ_ONLY_MESSAGE)
    if agent_crud.has_active_project_run(session, project_id):
        raise AgentArtifactConflictError(ACTIVE_RUN_MESSAGE)

    path, kind, run = _resolve_artifact(session, project_id, owner_id, artifact_id)
    if _revision(path) != expected_revision:
        raise AgentArtifactConflictError(STALE_REVISION_MESSAGE)
    if not content.strip():
        raise AgentArtifactValidationError(EMPTY_PLAN_MESSAGE)
    encoded = content.encode('utf-8')
    if len(encoded) > MAX_EDIT_BYTES:
        raise AgentArtifactValidationError(PLAN_TOO_LARGE_MESSAGE)

    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return _metadata(session, project_id, path, kind, run, active_run=False)
