from pathlib import Path

from app.schemas.workflow_config import ToolOutput


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


def missing_output_paths(output: ToolOutput) -> list[Path]:
    """Return declared tool outputs that have no exact or prefixed artifact."""
    return [Path(path) for path in output.all_files().values() if not output_path_exists(path)]


def all_output_paths_exist(output: ToolOutput) -> bool:
    """Return whether every declared tool output has an exact or prefixed artifact."""
    return not missing_output_paths(output)
