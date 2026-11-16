"""Helpers for planning and resolving deterministic container mount paths."""

import os
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from loguru import logger
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

import bioflow_ai.crud.project_file_mapping as project_file_mapping_crud
import bioflow_ai.crud.resource.sample as sample_crud
from bioflow_ai.infrastructure.containers import ContainerMount
from bioflow_ai.infrastructure.database import engine

PROJECT_SAMPLES_ROOT = '/data/project_samples'
PROJECT_REFERENCE_ROOT = '/data/project_reference'
PROJECT_DATA_ROOTS = (PROJECT_SAMPLES_ROOT, PROJECT_REFERENCE_ROOT)


def build_uv_mounts() -> list[ContainerMount]:
    """Mount host uv storage at the container root user's default locations."""
    user_home = Path.home()
    cache_home = Path(os.environ.get('XDG_CACHE_HOME') or user_home / '.cache')
    data_home = Path(os.environ.get('XDG_DATA_HOME') or user_home / '.local' / 'share')
    locations = [
        (os.environ.get('UV_CACHE_DIR') or cache_home / 'uv', '/root/.cache/uv'),
        (os.environ.get('UV_PYTHON_INSTALL_DIR') or data_home / 'uv' / 'python', '/root/.local/share/uv/python'),
    ]
    mounts = []
    for source, target in locations:
        directory = Path(source).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        mounts.append(ContainerMount(source=str(directory), target=target, read_only=False))
        if target == '/root/.local/share/uv/python' and str(directory) != target:
            # uv's managed Python aliases can contain absolute host paths.
            mounts.append(ContainerMount(source=str(directory), target=str(directory), read_only=False))
    return mounts


def resolve_recorded_input_path(
    container_path: str, mounts: Iterable[ContainerMount], *, for_copy: bool = False
) -> Path:
    """Resolve an input through the most specific recorded mount, including fixture mounts."""
    path = PurePosixPath(container_path)
    if not path.is_absolute() or '..' in path.parts:
        msg = f'Path must be absolute and cannot contain parent traversal: {container_path}'
        raise ValueError(msg)
    latest = {mount.target: mount for mount in mounts}
    for mount in sorted(latest.values(), key=lambda item: len(PurePosixPath(item.target).parts), reverse=True):
        target = PurePosixPath(mount.target)
        if path.is_relative_to(target):
            source = Path(mount.source).expanduser().resolve()
            mounted_path = source / Path(*path.relative_to(target).parts)
            return mounted_path if for_copy else mounted_path.resolve()
    msg = f'Input path is not available through a recorded mount: {container_path}'
    raise ValueError(msg)


class ProjectDataMountPlan(BaseModel):
    """Deterministic container mounts for the registered files of one project."""

    model_config = ConfigDict(frozen=True)

    mounts: tuple[ContainerMount, ...] = ()

    def resolve_input_path(self, container_path: str, *, for_copy: bool = False) -> Path:
        """Translate one concrete managed container input path back to its host path."""
        path = PurePosixPath(container_path)
        if not path.is_absolute() or '..' in path.parts:
            msg = f'Path must be absolute and cannot contain parent traversal: {container_path}'
            raise ValueError(msg)

        ordered_mounts = sorted(
            self.mounts,
            key=lambda mount: len(PurePosixPath(mount.target).parts),
            reverse=True,
        )
        for mount in ordered_mounts:
            target = PurePosixPath(mount.target)
            try:
                relative = path.relative_to(target)
            except ValueError:
                continue
            source = Path(mount.source)
            host_path = (source / Path(*relative.parts)).absolute()
            if for_copy and not host_path.resolve().is_relative_to(source.resolve()):
                msg = f'Input path resolves outside its project mount: {container_path}'
                raise ValueError(msg)
            return host_path

        msg = f'Path is not available through a managed project mount: {container_path}'
        raise ValueError(msg)


def resolve_mount_path(
    mount_paths: dict[str, Path],
    param_inputs: dict[str, str],
    conflicted_paths: set[str],
    container_mount_path: str,
    host_mount_path: Path,
) -> str:
    """Assign a unique container path when host directory mounts conflict."""
    abs_host = host_mount_path.absolute()
    existing = mount_paths.get(container_mount_path)

    if existing is None and container_mount_path not in conflicted_paths:
        return container_mount_path

    if existing == abs_host:
        return container_mount_path

    if existing is not None and container_mount_path not in conflicted_paths:
        conflicted_paths.add(container_mount_path)
        old_host = mount_paths.pop(container_mount_path)
        old_sub = f'{container_mount_path}/{old_host.name}'
        mount_paths[old_sub] = old_host
        for key, value in param_inputs.items():
            if value == container_mount_path or value.startswith(container_mount_path + '/'):
                relative = value[len(container_mount_path) :]
                param_inputs[key] = old_sub + relative

    candidate = f'{container_mount_path}/{host_mount_path.name}'
    counter = 1
    while candidate in mount_paths and mount_paths[candidate] != abs_host:
        candidate = f'{container_mount_path}/{host_mount_path.name}_{counter}'
        counter += 1

    logger.warning(f'挂载冲突: {container_mount_path} 已分裂, 将 {abs_host} 映射到 {candidate}')
    return candidate


def _source_roots(paths: Iterable[str], *, kind: str) -> list[Path]:
    roots: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().absolute()
        try:
            exists = path.exists()
            is_dir = path.is_dir()
            is_file = path.is_file()
        except OSError as error:
            logger.warning(f'Skipping inaccessible {kind} path {path}: {error}')
            continue
        if not exists or not (is_dir or is_file):
            logger.warning(f'Skipping missing or unsupported {kind} path: {path}')
            continue

        source_root = path if is_dir else path.parent
        if source_root == Path(source_root.anchor):
            logger.warning(f'Skipping unsafe {kind} path whose source root is the filesystem root: {path}')
            continue
        roots.add(source_root)
    return sorted(roots, key=str)


def _plan_root_mounts(paths: Iterable[str], container_root: str, *, kind: str) -> list[ContainerMount]:
    source_roots = _source_roots(paths, kind=kind)
    if not source_roots:
        return []
    if len(source_roots) == 1:
        return [ContainerMount(source=str(source_roots[0]), target=container_root, read_only=True)]

    mount_paths: dict[str, Path] = {}
    conflicted_paths: set[str] = set()
    for source_root in source_roots:
        target = resolve_mount_path(mount_paths, {}, conflicted_paths, container_root, source_root)
        if target not in mount_paths:
            mount_paths[target] = source_root

    return [
        ContainerMount(source=str(source), target=target, read_only=True)
        for target, source in sorted(mount_paths.items())
    ]


def plan_project_data_mounts(
    *,
    sample_paths: Iterable[str],
    reference_paths: Iterable[str],
) -> ProjectDataMountPlan:
    """Build fixed project sample and reference mounts from registered paths."""
    mounts = [
        *_plan_root_mounts(sample_paths, PROJECT_SAMPLES_ROOT, kind='project sample'),
        *_plan_root_mounts(reference_paths, PROJECT_REFERENCE_ROOT, kind='project reference'),
    ]
    return ProjectDataMountPlan(mounts=tuple(mounts))


def build_project_data_mount_plan(project_id: int) -> ProjectDataMountPlan:
    """Load non-dynamic project paths and build their read-only mount plan."""
    with Session(engine) as session:
        sample_files = sample_crud.get_files_by_project(session, project_id)
        project_files = project_file_mapping_crud.get_all_by_project(session, project_id)

    return plan_project_data_mounts(
        sample_paths=(
            sample_file.file_path for sample_file, _sample_name in sample_files if not sample_file.is_dynamic
        ),
        reference_paths=(project_file.file_path for project_file in project_files if not project_file.is_dynamic),
    )
