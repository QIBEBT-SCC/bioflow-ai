"""Isolated input copies for testing generated pass-through Tool configurations."""

import hashlib
from pathlib import Path
from shutil import copy2, copytree
from typing import TYPE_CHECKING

from bioflow_ai.services.workflow.validation.passthrough import validate_managed_paths, validate_workspace_inputs

if TYPE_CHECKING:
    from bioflow_ai.services.tools.execution import DockerToolExecutionConfig, ResolvedToolInput


def copy_path(source: Path, destination: Path) -> Path:
    """Copy a file or directory, following symlinks so the copy shares no data with its source."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        copytree(source, destination, dirs_exist_ok=True)
    else:
        copy2(source, destination)
    return destination


def copy_validation_paths(paths: dict[str, Path], output_root: Path) -> dict[str, Path]:
    """Isolate mutable test inputs while keeping files that share a parent (such as companions) together."""
    directory_roots = sorted((path for path in paths.values() if path.is_dir()), key=lambda path: len(path.parts))
    copies: dict[str, Path] = {}
    for name, source in sorted(paths.items(), key=lambda item: len(item[1].parts)):
        parent = next((root for root in directory_roots if source.is_relative_to(root)), source.parent)
        group = hashlib.sha256(str(parent).encode()).hexdigest()[:24]
        destination = output_root / '_validation_inputs' / group / source.relative_to(parent)
        try:
            copies[name] = destination if destination.exists() else copy_path(source, destination)
        except OSError as error:
            message = f'Could not copy INPUT {name!r} from {source} into the validation directory. {error}'
            raise OSError(message) from error
    return copies


def isolate_validation_inputs(
    config: 'DockerToolExecutionConfig',
    inputs: dict[str, 'ResolvedToolInput'],
    attempt_root: Path,
    managed_root: Path,
) -> dict[str, 'ResolvedToolInput']:
    """Fork writable inputs and inherited mounts without changing their container paths."""
    workspace = validate_workspace_inputs(
        config.file_mounts, inputs, config.tool_type, managed_root.name, str(managed_root)
    )
    sources = [Path(source) for source in workspace.mounts.values()] if workspace else []
    for file in config.file_mounts:
        if file.is_passthrough:
            path = Path(inputs[file.name].value)
            sources.append(path if path.is_dir() else path.parent)
    sources = validate_managed_paths(sources, managed_root)
    copies: dict[Path, Path] = {}
    for source in sorted(set(sources), key=lambda path: (len(path.parts), str(path))):
        if source.is_relative_to(attempt_root):
            continue
        if attempt_root.is_relative_to(source):
            message = f'Cannot copy a validation input containing the current attempt: {source}'
            raise ValueError(message)
        if not any(source.is_relative_to(parent) for parent in copies):
            copies[source] = copy_path(source, attempt_root / 'inputs' / str(len(copies) + 1))

    def remap(value: str) -> str:
        path = Path(value).resolve()
        for source, destination in copies.items():
            if path.is_relative_to(source):
                return str(destination / path.relative_to(source))
        return value

    fork = (
        workspace.model_copy(
            update={
                'path': remap(workspace.path),
                'mounts': {target: remap(source) for target, source in workspace.mounts.items()},
            }
        )
        if workspace
        else None
    )
    literal_inputs = {file.name for file in config.file_mounts if file.file_path.startswith('value:')}
    return {
        name: value.model_copy(
            update={
                'value': value.value if name in literal_inputs else remap(value.value),
                'workspace': fork if value.workspace is not None else None,
            }
        )
        for name, value in inputs.items()
    }
