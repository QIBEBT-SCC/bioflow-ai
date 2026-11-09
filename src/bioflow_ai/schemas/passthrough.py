"""The shared execution directory and original input mounts of a Tool chain."""

from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field


class ToolWorkspace(BaseModel):
    """One workspace inherited by successive DOCKER_PASSTHROUGH tools."""

    model_config = ConfigDict(frozen=True)

    path: str
    run_uid: str
    mounts: dict[str, str] = Field(default_factory=dict)


def find_binding(mounts: Mapping[str, str], value: str | Path) -> tuple[str, str, str] | None:
    """Return ``(target, source, container_path)`` for the deepest mount source containing ``value``."""
    host = Path(value).absolute()
    for target, source in sorted(mounts.items(), key=lambda item: len(Path(item[1]).parts), reverse=True):
        if host.is_relative_to(source):
            return target, source, str(PurePosixPath(target) / host.relative_to(source).as_posix())
    return None


def inherited_log_path(workspace: str | Path, node_dir: str | Path) -> Path:
    """Per-node log of a pass-through Tool running in an inherited workspace.

    ``node_dir`` is the node's own compiled output directory (``{position}_{tool}``), unique within a run,
    so successive Tools sharing the workspace never overwrite each other's ``11_run.log``.
    """
    return Path(workspace) / f'{Path(node_dir).name}.log'
