"""Sample and project-file registration tools for the project-planner agent."""

import uuid

from anyio import Path
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import bioflow_ai.crud.project_file_mapping as project_file_mapping_crud
import bioflow_ai.crud.resource.sample as sample_crud
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.llm.context import ProjectContext
from bioflow_ai.llm.states import ProjectPlannerState
from bioflow_ai.models.resource import Sample, SampleFiles
from bioflow_ai.services.projects import file_mapping as project_file_mapping_service
from bioflow_ai.services.resource import sample as sample_service


@tool(parse_docstring=True)
async def add_samples(
    names: list[str],
    meta_data: dict[str, str] | None,
    runtime: ToolRuntime[ProjectContext, ProjectPlannerState],
) -> Command | str:
    """Create sample objects in batch; all samples in one call share the same metadata.

    Args:
        names: Unique sample names within the project.
        meta_data: Confirmed input metadata shared by this batch; omit unknown fields.
    """
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        return f'Duplicate sample name(s) in this batch: {duplicates}. Sample names must be unique, please rename and retry.'

    samples = [
        Sample(
            owner_id=runtime.context.user_id,
            project_id=runtime.context.project_id,
            sample_name=n,
            meta_data=meta_data or {},
        )
        for n in names
    ]
    with Session(engine) as session:
        try:
            samples = sample_service.create_samples(session, samples)
        except IntegrityError:
            session.rollback()
            return (
                'One or more sample names already exist in this project. '
                'Call `list_project_samples` to check existing names, then retry with unique ones.'
            )
        sample_maps = {s.sample_name: f'{s.uid}' for s in samples}

    return Command(
        update={
            'sample_maps': sample_maps,
            'messages': [
                ToolMessage(
                    content=f'Added {len(names)} sample objects. Now you can call `add_sample_file` to add files.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(parse_docstring=True)
async def add_sample_file(
    sample_name: str,
    tag: str,
    file_path: str,
    runtime: ToolRuntime[ProjectContext, ProjectPlannerState],
) -> str:
    """Register a file or directory for an existing sample.

    Args:
        sample_name: Target sample name.
        file_path: File or directory path relative to `base_path`.
        tag: Unique file tag within the sample, such as `r1`, `r2`, `read`, `spectrum`, or `img`.
    """
    base_dir = runtime.state.get('base_path')
    if not base_dir:
        return 'Call `set_base_dir` to set the root directory first.'

    sample_path = Path(base_dir) / file_path
    if not await sample_path.exists():
        return f'The sample file does not exist. The path constructed by concatenating base_dir and the provided file_path is: {sample_path}. Please verify that the path is correct.'

    sample_maps: dict[str, str] = runtime.state.get('sample_maps', {})
    if not sample_maps:
        return 'Call `add_samples` to create sample objects first.'

    if sample_uid := sample_maps.get(sample_name):
        with Session(engine) as session:
            sample_file = SampleFiles(
                sample_uid=uuid.UUID(sample_uid),
                file_path=f'{sample_path}',
                tag=tag,
            )
            try:
                sample_file = sample_service.add_file(session, sample_file)
            except IntegrityError:
                session.rollback()
                return f'Tag "{tag}" already exists for sample {sample_name}. Choose a different tag.'

        return f'Add "{sample_file.tag}" file for sample {sample_name}.'

    return f'No sample object named {sample_name} is found. Please ensure that sample_name is correct.'


@tool(parse_docstring=True)
async def add_project_file(
    keyword: str,
    file_path: str,
    description: str,
    runtime: ToolRuntime[ProjectContext, ProjectPlannerState],
) -> str:
    """Register a project-level shared file or directory.

    Args:
        keyword: Unique project-level keyword.
        file_path: File or directory path relative to `base_path`.
        description: Factual description of the file's role and format characteristics.
    """
    base_dir = runtime.state.get('base_path')
    if not base_dir:
        return 'Call `set_base_dir` to set the root directory first.'

    final_path = Path(base_dir) / file_path
    if not await final_path.exists():
        return f'The sample file does not exist. The path constructed by concatenating base_dir and the provided file_path is: {final_path}. Please verify that the path is correct.'

    with Session(engine) as session:
        try:
            project_file_mapping_service.create_mapping(
                session,
                runtime.context.project_id,
                keyword,
                f'{final_path}',
                description,
            )
        except IntegrityError:
            session.rollback()
            return f'Project file keyword "{keyword}" already exists. Choose a different keyword.'
    return f'Added project file "{keyword}"'


@tool(parse_docstring=True)
async def list_project_samples(runtime: ToolRuntime[ProjectContext, ProjectPlannerState]) -> str:
    """Summarize samples and project-level files currently registered in this project's database, grouped by metadata and tags."""
    project_id = runtime.context.project_id
    with Session(engine) as session:
        groups = sample_crud.summarize_samples_by_project(session, project_id)
        proj_files = project_file_mapping_crud.get_all_by_project(session, project_id)

    if not groups and not proj_files:
        return 'No samples or project-level files are registered for this project. Proceed with filesystem scan.'

    lines: list[str] = []
    total = sum(g.count for g in groups)
    lines.append(f'## Existing registry\nTotal samples: {total}')

    for idx, g in enumerate(groups, start=1):
        lines.append(f'\n### Group {idx} — {g.count} sample(s)')
        lines.append(f'- meta: {g.meta_data or "{}"}')
        if g.tags:
            tag_str = ', '.join(g.tags)
            lines.append(f'- tags: {tag_str}')
        else:
            lines.append('- tags: (none registered)')
        lines.append(f'- examples: {", ".join(g.sample_name_examples)}')

    if proj_files:
        lines.append('\n### Project-level files')
        lines.extend(f'- proj:{pf.keyword} — {pf.description}' for pf in proj_files)

    return '\n'.join(lines)
