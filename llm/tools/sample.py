import uuid
from pathlib import Path
from typing import Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from sqlmodel import Session

import app.crud.project as proj_crud
import app.crud.resource.sample as sample_crud
from app.core.session import engine
from app.models.resource import Sample, SampleFiles, SampleFileType
from app.models.tools import ImageConfig
from app.utils.container import ContainerMount, run_command
from llm.states import SampleContext

TYPE_MAP = {
    'SEQUENCING_R1': SampleFileType.SEQUENCING_R1,
    'SEQUENCING_R2': SampleFileType.SEQUENCING_R2,
    'SEQUENCING': SampleFileType.SEQUENCING,
    'SPECTRUM': SampleFileType.SPECTRUM,
    'IMAGE': SampleFileType.IMAGE,
}


def _truncate_output(output: str) -> str:
    if len(output) <= 3000:  # noqa: PLR2004
        return output
    return f'{output[:1500]}\n...\n{output[-500:]}'


@tool(parse_docstring=True)
async def set_base_dir(base_path: str, runtime: ToolRuntime) -> Command:
    """Set the root directory.

    After setting, all subsequent operations (including run_command and all register-type tools) use this as the working directory.
    This should be called again each time the dataset is switched.

    Args:
        base_path: Root directory of the currently operating
    """
    return Command(
        update={
            'base_path': base_path,
            'messages': [
                ToolMessage(
                    content=f'The root directory has been set to {base_path}', tool_call_id=runtime.tool_call_id
                )
            ],
        }
    )


@tool('run_command', parse_docstring=True)
async def run_command_tool(command: str, runtime: ToolRuntime[SampleContext]) -> str:
    """Execute shell commands inside a Docker container.

    Available commands: seqkit, awk, grep, find, head, file, etc.
    The configured root directory is mapped to the /app/data path of the container.

    Args:
        command: Shell command to run in the container
    """
    base_dir: str | None = runtime.state.get('base_path')
    if not base_dir:
        return 'Call `set_base_dir` to set the root directory first.'

    if not Path(base_dir).exists():
        return 'The root directory does not exist. Call the `set_base_dir` tool to set the correct root directory, or ask human to confirm the correct path.'

    run_image = ImageConfig.from_str('docker.io/aye1032/biokit-seq:1.0.0')
    file_mount = ContainerMount(source=base_dir, target='/app/data', read_only=True)
    code, result = run_command(run_image, [file_mount], command)

    run_status = 'successfully' if code == 0 else 'failed'
    if clean_out := result.strip():
        return_msg = (
            _truncate_output(clean_out)
            if code == 0
            else f'Command ran {run_status}.\n\n......\n{_truncate_output(clean_out)}'
        )
    else:
        return_msg = f'Command ran {run_status}. No other output.'
    return return_msg


@tool(parse_docstring=True)
async def add_samples(
    names: list[str],
    meta_data: dict[str, str] | None,
    runtime: ToolRuntime[SampleContext],
) -> Command:
    """Create sample objects.

    Samples added in the same batch share the same metadata.

    Args:
        names: The list of sample names. Each sample name must be unique and may use uppercase letters, hyphens, and numbers.
        meta_data: Meta information of the sample
    """
    samples = [
        Sample(
            owner_id=runtime.context.user_id,
            project_id=runtime.context.project_id,
            sample_name=n,
            meta_data=meta_data,
        )
        for n in names
    ]
    with Session(engine) as session:
        samples = sample_crud.batch_insert(session, samples)
        sample_maps = {s.sample_name: f'{s.uid}' for s in samples}

    return Command(
        update={
            'sample_maps': sample_maps,
            'messages': [
                ToolMessage(
                    content=f'Added {len(names)} sample objects. Now you can call `register_sample_file` to add sample files.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(parse_docstring=True)
async def register_sample_file(
    sample_name: str,
    data_type: Literal['SEQUENCING_R1', 'SEQUENCING_R2', 'SEQUENCING', 'SPECTRUM', 'IMAGE'],
    file_path: str,
    runtime: ToolRuntime[SampleContext],
    tag: str | None = None,
) -> str:
    """Register the sample file.

    Args:
        sample_name: The sample name of the file to add
        data_type: File type
        file_path: The file_path uses a path relative to **base_dir**.
        tag: When tag is not provided, the system uses a default value.
    """
    base_dir: str | None = runtime.state.get('base_path')
    if not base_dir:
        return 'Call `set_base_dir` to set the root directory first.'

    sample_path = Path(base_dir) / file_path
    if not sample_path.exists():
        return f'The sample file does not exist. The path constructed by concatenating base_dir and the provided file_path is: {sample_path}. Please verify that the path is correct.'

    sample_maps: dict[str, str] = runtime.state.get('sample_maps', {})
    if not sample_maps:
        return 'Call `add_samples` to create sample objects first.'

    if sample_uid := sample_maps.get(sample_name):
        with Session(engine) as session:
            sample_file = SampleFiles(
                sample_uid=uuid.UUID(sample_uid),
                data_type=TYPE_MAP[data_type],
                file_path=f'{sample_path}',
                tag=tag,
            )
            sample_file = sample_crud.add_file(session, sample_file)

        return f'Add "{sample_file.tag}" file for sample {sample_name}.'

    return f'No sample object named {sample_name} is found. Please ensure that sample_name is correct.'


@tool(parse_docstring=True)
async def register_project_file(
    keyword: str,
    file_path: str,
    description: str,
    runtime: ToolRuntime[SampleContext],
) -> str:
    """Register project-level shared files.

    Args:
        keyword: File key-word
        file_path: The file_path uses a path relative to **base_dir**.
        description: Description of this file.
    """
    base_dir: str | None = runtime.state.get('base_path')
    if not base_dir:
        return 'Call `set_base_dir` to set the root directory first.'

    final_path = Path(base_dir) / file_path
    if not final_path.exists():
        return f'The sample file does not exist. The path constructed by concatenating base_dir and the provided file_path is: {final_path}. Please verify that the path is correct.'

    with Session(engine) as session:
        proj_crud.create_file_mapping(
            session,
            runtime.context.project_id,
            keyword,
            f'{final_path}',
            description,
        )
    return f'Added project file "{keyword}"'


@tool(parse_docstring=True)
async def list_project_samples(runtime: ToolRuntime[SampleContext]) -> str:
    """List existing samples already registered in the database for the current project.

    Returns an aggregated summary grouped by shared metadata and tag set, suitable for projects
    with hundreds of samples. Call this BEFORE any filesystem exploration to decide whether
    re-scanning is needed.
    """
    project_id = runtime.context.project_id
    with Session(engine) as session:
        groups = sample_crud.summarize_samples_by_project(session, project_id)
        proj_files = proj_crud.get_all_file_mappings(session, project_id)

    if not groups and not proj_files:
        return 'No samples or project-level files are registered for this project. Proceed with filesystem scan.'

    lines: list[str] = []
    total = sum(g.count for g in groups)
    lines.append(f'## Existing registry\nTotal samples: {total}')

    for idx, g in enumerate(groups, start=1):
        lines.append(f'\n### Group {idx} — {g.count} sample(s)')
        lines.append(f'- meta: {g.meta_data or "{}"}')
        if g.tags:
            tag_str = ', '.join(f'{t.tag} ({t.data_type.name})' for t in g.tags)
            lines.append(f'- tags: {tag_str}')
        else:
            lines.append('- tags: (none registered)')
        lines.append(f'- examples: {", ".join(g.sample_name_examples)}')

    if proj_files:
        lines.append('\n### Project-level files')
        lines.extend(f'- proj:{pf.keyword} — {pf.description}' for pf in proj_files)

    return '\n'.join(lines)


@tool('remove_sample_file', parse_docstring=True)
async def remove_sample_file(sample_name: str, tag: str, runtime: ToolRuntime[SampleContext]) -> str:
    """Remove a registered sample file by sample name and tag.

    Use this only on the "local files updated" branch to drop stale entries before
    re-registering them. Does not delete the sample itself.

    Args:
        sample_name: Name of the sample whose file should be removed.
        tag: Tag of the file to remove (e.g. 'r1', 'r2').
    """
    with Session(engine) as session:
        deleted = sample_crud.delete_file_by_name_tag(session, runtime.context.project_id, sample_name, tag)
    if deleted:
        return f'Removed file "{tag}" of sample {sample_name}.'
    return f'No file with tag "{tag}" found on sample {sample_name}.'


@tool
async def write_conclusion(content: str, runtime: ToolRuntime[SampleContext]) -> Command:
    """Write a summary for the file system."""
    return Command(
        update={
            'summary': content,
            'updated': True,
            'messages': [ToolMessage(content='Done.', tool_call_id=runtime.tool_call_id)],
        }
    )
