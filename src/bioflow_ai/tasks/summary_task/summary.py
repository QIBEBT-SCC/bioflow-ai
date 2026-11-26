"""Explicit downstream-summary node executor."""

import shutil
from pathlib import Path

from loguru import logger

from bioflow_ai.schemas.react_flow import TaskInput
from bioflow_ai.schemas.workflow_config import ToolOutput
from bioflow_ai.services.workflow.execution_context import get_node_output


def build_downstream_summary(
    *,
    inputs: dict[str, TaskInput | list[TaskInput]],
    prompt: str,
    output_dir: str,
) -> ToolOutput:
    """Collect downstream-analysis files and materialize the explicit coding prompt."""
    input_files = inputs.get('input_files', [])
    references = input_files if isinstance(input_files, list) else [input_files]

    output_folder = Path(output_dir)
    if output_folder.is_dir() and not output_folder.is_symlink():
        shutil.rmtree(output_folder)
    else:
        output_folder.unlink(missing_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    for reference in references:
        origin_file = Path(get_node_output(reference.source_uid, reference.source_k)[0])
        if origin_file.exists():
            new_file = output_folder / origin_file.name
            if new_file.is_symlink() or new_file.is_file():
                new_file.unlink()
            elif new_file.is_dir():
                shutil.rmtree(new_file)
            new_file.symlink_to(origin_file)
        else:
            logger.warning(f'file {origin_file} did not exist!')

    claude_command = f'claude --dangerously-skip-permissions --output-format=stream-json --verbose -p "{prompt}"'
    claude_bash = output_folder.parent / 'run_downstream.sh'
    claude_bash.unlink(missing_ok=True)
    claude_bash.write_text(claude_command)

    return ToolOutput(result={})
