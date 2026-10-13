from collections.abc import Collection
from pathlib import Path

from bioflow_ai.schemas.workflow_config import ToolOutput


def output_path_exists(path: str | Path) -> bool:
    """Return whether an exact output path or a matching prefixed artifact exists."""
    output_path = Path(path)
    if output_path.exists():
        return True

    try:
        return any(
            candidate.name != output_path.name and candidate.name.startswith(output_path.name)
            for candidate in output_path.parent.iterdir()
        )
    except (FileNotFoundError, NotADirectoryError):
        return False


def missing_output_paths(
    output: ToolOutput,
    *,
    required_output_names: Collection[str] | None = None,
) -> list[Path]:
    """Return missing required paths; omitted requirements preserve checking every output."""
    return [
        Path(path)
        for name, path in output.all_files().items()
        if (required_output_names is None or name in required_output_names) and not output_path_exists(path)
    ]


def all_output_paths_exist(
    output: ToolOutput,
    *,
    required_output_names: Collection[str] | None = None,
) -> bool:
    """Return whether every required output has an exact or prefixed artifact."""
    return not missing_output_paths(output, required_output_names=required_output_names)
