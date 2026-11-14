"""Configuration, topology, and runtime checks for inherited Tool workspaces."""

from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from bioflow_ai.schemas.passthrough import ToolWorkspace
from bioflow_ai.schemas.react_flow import WorkflowDefinition
from bioflow_ai.schemas.tool_config import FileMount, ToolType
from bioflow_ai.schemas.tools import WorkflowToolDefinition
from bioflow_ai.schemas.workflow_config import ToolOutput
from bioflow_ai.services.workflow.validation.types import Diagnostic

if TYPE_CHECKING:
    import uuid

    from bioflow_ai.models.workflow import NodeRun
    from bioflow_ai.services.tools.execution import ResolvedToolInput

PATH_INPUTS = {'select_file_in_folder': 'folder_path', 'join_path': 'base_path'}
COLLECTIONS = {'collect_mount_dirs', 'collect_file_collection', 'foreach'}


def _normalized_data_path(target: str) -> bool:
    path = PurePosixPath(target)
    return target.startswith('/data/') and '..' not in path.parts and str(path) == target


def tool_mount_diagnostics(files: list[FileMount], tool_type: ToolType) -> list[Diagnostic]:
    """Report Tool file-mount rules without validating or mutating a schema model."""
    diagnostics = []

    def report(message: str, path: str, handle: str | None = None) -> None:
        diagnostics.append(Diagnostic(code='passthrough.configuration', message=message, path=path, handle=handle))

    for name, count in Counter(file.name for file in files).items():
        if count > 1:
            report(
                f'File mount names must be unique, including automatic pass-through outputs: {name}',
                '/file_mounts',
                name,
            )
    if tool_type != ToolType.DOCKER_PASSTHROUGH and any(file.is_passthrough for file in files):
        report('Pass-through inputs require tool_type=1 (DOCKER_PASSTHROUGH)', '/tool_type')
    for index, file in enumerate(files):
        if not file.is_passthrough:
            continue
        path = f'/file_mounts/{index}'
        if file.name in {'workdir', 'log'}:
            report('Pass-through handles cannot use reserved names workdir or log', path + '/name', file.name)
        if file.file_type != 'INPUT' or file.file_path != f'passthrough:{{{file.name}}}':
            report('Pass-through requires an INPUT with file_path="passthrough:{name}"', path + '/file_path', file.name)
        if not _normalized_data_path(file.mount_path):
            report(
                'Pass-through input mounts must be normalized absolute paths under /data',
                path + '/mount_path',
                file.name,
            )
        if file.is_log or file.is_report:
            report('Pass-through handles cannot be logs or reports', path, file.name)
    return diagnostics


def validate_file_mounts(files: list[FileMount], tool_type: ToolType) -> None:
    """Apply the same configuration rules at Tool save and execution entry points."""
    if issues := tool_mount_diagnostics(files, tool_type):
        message = '\n'.join(f'{issue.path}: {issue.message}' for issue in issues)
        raise ValueError(message)


def _external_source(node_type: str) -> bool:
    return (node_type.startswith('resource_') and node_type != 'resource_text_file') or node_type == 'value_string'


def _requires_copy_for_passthrough(
    node_type: str,
    data: Mapping[str, Any],
    *,
    source_handle: str = '',
    dynamic_marks: Collection[str] = (),
) -> bool:
    if node_type in {'resource_text_file', 'resource_frozen_path'}:
        return False
    if node_type == 'resource_file':
        return str(data.get('file_path', '')).strip() not in dynamic_marks
    if node_type == 'resource_sequence':
        key = source_handle or 'r1'
        default = f'sample:{key}' if key in {'r1', 'r2'} else ''
        return str(data.get(key, default)).strip() not in dynamic_marks
    return _external_source(node_type)


def _exports_output_workspace(tool: WorkflowToolDefinition | None, handle: str) -> bool:
    if not (tool and tool.tool_type == ToolType.DOCKER_PASSTHROUGH):
        return False
    mount = next((item for item in tool.file_mounts if item.name == handle), None)
    if mount is None:
        return handle == 'workdir'
    return PurePosixPath(mount.mount_path) == PurePosixPath('/data/output') and (
        mount.file_type == 'OUTPUT' or mount.is_passthrough
    )


def passthrough_diagnostics(
    graph: WorkflowDefinition,
    catalog: dict[str, WorkflowToolDefinition],
    *,
    dynamic_marks: Collection[str] = (),
) -> list[Diagnostic]:
    """Keep shared ``/data/output`` workspace chains linear while allowing direct workflow and dynamic-mark inputs.

    A pass-through Tool accepts at most one shared ``/data/output`` workspace predecessor, and a
    workspace predecessor continues into at most one pass-through Tool. Non-workspace sidecar or
    in-place handles (outside ``/data/output``), dynamic project/sample marks, and existing
    workflow node outputs connect directly without ``copy2folder``.
    """
    nodes = {node.id: node for node in graph.nodes}
    incoming: defaultdict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for edge in graph.edges:
        incoming[edge.target].append(
            (
                edge.source,
                edge.sourceHandle.removeprefix(f'{edge.source}-out-'),
                edge.targetHandle.removeprefix(f'{edge.target}-in-'),
            )
        )
    tools = {node.id: catalog.get(str(node.data.get('tool_uid'))) for node in graph.nodes if node.type == 'tool'}
    diagnostics = []

    def report(uid: str, message: str) -> None:
        diagnostics.append(Diagnostic(code='passthrough.contract', node_id=uid, message=message))

    def origins(uid: str, source_handle: str = '', seen: frozenset[str] = frozenset()) -> set[tuple[str, str]]:
        if uid in seen or uid not in nodes:
            return set()
        if base_input := PATH_INPUTS.get(nodes[uid].type):
            return set().union(
                *(origins(src, src_h, seen | {uid}) for src, src_h, tgt_h in incoming[uid] if tgt_h == base_input)
            )
        return {(uid, source_handle)}

    def shared_sources(uid: str, source_handle: str = '') -> set[str]:
        return {
            origin_uid
            for origin_uid, origin_handle in origins(uid, source_handle)
            if _exports_output_workspace(tools.get(origin_uid), origin_handle)
        }

    consumers: defaultdict[str, set[str]] = defaultdict(set)
    for node in graph.nodes:
        tool = tools.get(node.id)
        if tool and tool.tool_type == ToolType.DOCKER_PASSTHROUGH:
            pass_inputs = {file.name for file in tool.file_mounts if file.is_passthrough}
            sources = set().union(
                *(shared_sources(src, src_h) for src, src_h, tgt_h in incoming[node.id] if tgt_h in pass_inputs)
            )
            for source in sources:
                consumers[source].add(node.id)
            if len(sources) > 1:
                report(
                    node.id,
                    'A pass-through tool accepts at most one shared workspace source node: '
                    + ', '.join(sorted(sources)),
                )
            for src, src_h, tgt_h in incoming[node.id]:
                if tgt_h in pass_inputs:
                    for origin_uid, origin_handle in origins(src, src_h):
                        producer = nodes[origin_uid]
                        if _requires_copy_for_passthrough(
                            producer.type,
                            producer.data,
                            source_handle=origin_handle,
                            dynamic_marks=dynamic_marks,
                        ):
                            report(
                                node.id,
                                'Initial project/sample resources and external files must be copied '
                                '(e.g. via copy2folder and select_file_in_folder) before connecting to '
                                f'pass-through input {tgt_h!r}',
                            )
        if node.type in COLLECTIONS:
            sources = set().union(*(shared_sources(src, src_h) for src, src_h, _ in incoming[node.id]))
            if sources:
                report(node.id, 'Pass-through workspace cannot be exported through collections or ForEach')
    for source, targets in sorted(consumers.items()):
        if len(targets) > 1:
            for target in sorted(targets):
                others = ', '.join(sorted(targets - {target}))
                report(
                    target,
                    f'Pass-through workspace source {source} can continue into only one pass-through tool; '
                    f'also used by {others}',
                )
    return diagnostics


def validate_managed_paths(
    required: Iterable[str | Path],
    managed_root: str | Path,
    *,
    optional: Iterable[str | Path] = (),
    allowed_roots: Iterable[str | Path] = (),
) -> list[Path]:
    """Reject paths and links escaping managed workflow storage.

    Every directory tree is scanned for symlinks once, even when several paths share it.

    Args:
        required: Paths that must exist as regular files or directories.
        managed_root: The current workflow run directory.
        optional: Paths that may not exist yet, such as optional outputs.
        allowed_roots: Verified dynamic-mark producer directories also trusted for pass-through access.

    Returns:
        The resolved required paths, in input order.
    """
    roots = [Path(managed_root).resolve(strict=True), *(Path(item).resolve(strict=True) for item in allowed_roots)]
    required_paths = [Path(value).absolute() for value in required]
    optional_paths = [Path(value).absolute() for value in optional]
    for path in [*required_paths, *optional_paths]:
        resolved = path.resolve()
        if not any(resolved.is_relative_to(root) for root in roots):
            message = f'Pass-through requires a workflow-produced copy: {path}'
            raise ValueError(message)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if not (path.is_file() or path.is_dir()):
            message = f'Pass-through requires a regular file or directory: {path}'
            raise ValueError(message)
    scanned: list[Path] = []
    for path in sorted({*required_paths, *optional_paths}, key=lambda item: len(item.parts)):
        if any(path.is_relative_to(tree) for tree in scanned):
            continue
        for entry in [path, *(path.rglob('*') if path.is_dir() else [])]:
            if entry.is_symlink() and (
                not entry.exists() or not any(entry.resolve().is_relative_to(root) for root in roots)
            ):
                message = f'External or dangling symlink in pass-through data: {entry}; copy the data first'
                raise ValueError(message)
        if path.is_dir():
            scanned.append(path)
    return [path.resolve() for path in required_paths]


def check_mount_conflicts(mounts: Mapping[str, str | Path], fixed: Mapping[str, str | Path]) -> None:
    """Reject both exact collisions and parent/child masking of fixed bindings."""
    for target, source in fixed.items():
        for other_target, other_source in mounts.items():
            a, b = PurePosixPath(target), PurePosixPath(other_target)
            if a == b and Path(source) == Path(other_source):
                continue
            if a == b or a.is_relative_to(b) or b.is_relative_to(a):
                message = f'Fixed mount conflict: {source} at {target} and {other_source} at {other_target}'
                raise ValueError(message)


def validate_workspace(
    workspace: ToolWorkspace,
    run_uid: str,
    managed_root: str,
    *,
    outputs: Iterable[str | Path] = (),
    allowed_roots: Iterable[str | Path] = (),
) -> None:
    """Recheck the shared directory, its input mounts, and optional result paths inside the current run."""
    if workspace.run_uid != run_uid:
        message = 'Pass-through cannot be shared across independent workflow runs'
        raise ValueError(message)
    path = Path(workspace.path)
    if not path.is_dir() or workspace.mounts.get('/data/output') != workspace.path:
        message = 'Pass-through workspace must be the original /data/output directory'
        raise ValueError(message)
    checked: dict[str, str] = {}
    for target, source in workspace.mounts.items():
        if not _normalized_data_path(target):
            message = f'Invalid workspace mount target: {target}'
            raise ValueError(message)
        check_mount_conflicts(checked, {target: source})
        checked[target] = source
    effective_roots = tuple(dict.fromkeys([*allowed_roots, *workspace.allowed_roots]))
    validate_managed_paths(workspace.mounts.values(), managed_root, optional=outputs, allowed_roots=effective_roots)


def validate_workspace_inputs(
    files: list[FileMount],
    inputs: Mapping[str, 'ResolvedToolInput'],
    tool_type: ToolType,
    run_uid: str,
    managed_root: str,
    *,
    allowed_roots: Iterable[str | Path] = (),
) -> ToolWorkspace | None:
    """Recheck source identities and input paths before execution.

    A tool inherits a shared ``/data/output`` workspace only through ``passthrough:`` inputs whose
    value resides inside that upstream workspace or targets ``/data/output``. Sidecar/in-place
    pass-through inputs outside ``/data/output`` may come from multiple upstream tools concurrently.

    Returns:
        The workspace inherited through a pass-through input, if any.
    """
    if tool_type != ToolType.DOCKER_PASSTHROUGH:
        return None
    if not run_uid or not managed_root:
        message = 'Pass-through tools require current workflow-run provenance'
        raise ValueError(message)

    passthrough = [inputs[file.name] for file in files if file.is_passthrough]
    if any(value.value.startswith(('mountlist:', 'value:')) for value in passthrough):
        message = 'Pass-through cannot consume value: or mountlist: inputs'
        raise ValueError(message)

    workspace_sources: dict[str, ToolWorkspace] = {}
    for file in files:
        if not file.is_passthrough:
            continue
        value = inputs[file.name]
        if value.workspace is None:
            continue
        if PurePosixPath(file.mount_path) == PurePosixPath('/data/output') or Path(
            value.value
        ).resolve().is_relative_to(Path(value.workspace.path).resolve()):
            workspace_sources[value.source_uid] = value.workspace

    if len(workspace_sources) > 1:
        message = 'A pass-through tool accepts at most one shared workspace source node'
        raise ValueError(message)

    effective_roots = tuple(
        dict.fromkeys(
            [
                *allowed_roots,
                *(value.managed_root for value in passthrough if value.managed_root),
                *(root for value in passthrough if value.workspace for root in value.workspace.allowed_roots),
            ]
        )
    )
    for workspace in workspace_sources.values():
        validate_workspace(workspace, run_uid, managed_root, allowed_roots=effective_roots)

    # A file outside the inherited workspace is mounted through its parent directory.
    hosts = [Path(value.value) for value in passthrough]
    validate_managed_paths(
        [*hosts, *(host if host.is_dir() else host.parent for host in hosts)],
        managed_root,
        allowed_roots=effective_roots,
    )
    return next(iter(workspace_sources.values()), None)


def validate_passthrough_origin(
    node: 'NodeRun',
    current: 'NodeRun | None',
    seen: 'set[uuid.UUID]',
    *,
    writable: bool = False,
    is_dynamic_mark: bool = False,
) -> None:
    """Check actual source records while the runtime resolves transparent path adapters."""
    if current is None or current.run_uid != node.run_uid or node.uid in seen:
        message = 'Invalid or cross-run pass-through source'
        raise ValueError(message)
    if writable and _external_source(node.node_type) and not is_dynamic_mark:
        message = 'Initial project/sample resources and external files must be copied before writable pass-through'
        raise ValueError(message)


def validate_workspace_outputs(
    output: ToolOutput, managed_root: str, *, allowed_roots: Iterable[str | Path] = ()
) -> None:
    """Check mutable results before a successful Tool publishes them."""
    if workspace := output.workspace:
        validate_workspace(
            workspace,
            workspace.run_uid,
            managed_root,
            outputs=output.all_files().values(),
            allowed_roots=allowed_roots,
        )
