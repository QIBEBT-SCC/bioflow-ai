import json
import re
import shutil
import uuid
from glob import has_magic
from pathlib import Path

from celery import Task
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.session import engine
from app.crud.resource import sample as sample_crud
from app.models.resource import Sample
from app.models.workflow import RunInstance
from app.schemas.react_flow import TaskInput
from app.schemas.workflow_config import ToolOutput
from app.services import project_file_mapping as project_file_mapping_service
from app.services import sample as sample_service
from app.services.output_validation import output_path_exists
from app.tasks.base.base_task import BaseBioTask, get_prefix, get_tool_output
from app.utils.input_format import to_safe_path_segment
from script.worker import celery_app

PROJECT_FILE_KEYWORD_MAX_LENGTH = 50
TEXT_FILE_CONTENT_MAX_BYTES = 5 * 1024 * 1024
MARK_NAME_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def create_text_file_task(
    self: Task,  # noqa: ARG001
    result_dict: ToolOutput | None = None,  # noqa: ARG001
    *,
    file_name: str,
    content: str,
    output_dir: str,
) -> ToolOutput:
    """Materialize inline text as a file for downstream workflow nodes."""
    if not file_name or file_name in {'.', '..'} or Path(file_name).name != file_name or '\\' in file_name:
        msg = 'file_name must be a single, non-empty file name'
        raise ValueError(msg)

    if len(content.encode('utf-8')) > TEXT_FILE_CONTENT_MAX_BYTES:
        msg = f'text file content must not exceed {TEXT_FILE_CONTENT_MAX_BYTES} bytes'
        raise ValueError(msg)

    output_folder = Path(output_dir).absolute()
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / file_name
    output_file.write_text(content, encoding='utf-8')
    return ToolOutput(result={'file_path': output_file})


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def select_file_task(
    self: Task,  # noqa: ARG001
    result_dict: ToolOutput | None = None,  # noqa: ARG001
    *,
    inputs: dict[str, TaskInput],
    file_name: str,
) -> ToolOutput:
    """从文件夹中提取指定文件."""
    kwarg = inputs.get('folder_path')
    if kwarg is None:
        raise ValueError
    parent_dir, sample_name = get_tool_output(kwarg.source_uid, kwarg.source_k)

    parent_path = Path(parent_dir)
    if has_magic(file_name):
        matched_files = sorted(parent_path.glob(file_name))
        if not matched_files:
            raise FileNotFoundError(parent_dir)
        selected_file = matched_files[0]
    else:
        selected_file = parent_path / file_name
        if not selected_file.exists():
            raise FileNotFoundError(parent_dir)

    return ToolOutput(result={'file_path': selected_file}, sample_name=sample_name)


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def rename_file_task(
    self: Task,  # noqa: ARG001
    result_dict: ToolOutput | None = None,  # noqa: ARG001
    *,
    inputs: dict[str, TaskInput],
    new_file_name: str,
    output_dir: str,
) -> ToolOutput:
    """Expose an upstream file under a new name without modifying it."""
    input_file = inputs.get('old_file')
    if input_file is None:
        msg = 'rename_file requires exactly one old_file input'
        raise ValueError(msg)

    if (
        not new_file_name
        or new_file_name in {'.', '..'}
        or Path(new_file_name).name != new_file_name
        or '\\' in new_file_name
    ):
        msg = 'new_file_name must be a single, non-empty file name'
        raise ValueError(msg)

    source_path_str, sample_name = get_tool_output(input_file.source_uid, input_file.source_k)
    source_path = Path(source_path_str).absolute()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not source_path.is_file():
        raise IsADirectoryError(source_path)

    output_folder = Path(output_dir).absolute()
    output_folder.mkdir(parents=True, exist_ok=True)
    renamed_file = output_folder / new_file_name
    if renamed_file.exists() or renamed_file.is_symlink():
        renamed_file.unlink()
    renamed_file.symlink_to(source_path)

    return ToolOutput(result={'renamed_file': renamed_file}, sample_name=sample_name)


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def collect_files_task(
    self: Task,  # noqa: ARG001
    result_dict: ToolOutput | None = None,  # noqa: ARG001
    *,
    inputs: dict[str, TaskInput | list[TaskInput]],
    output_dir: str,
) -> ToolOutput:
    """Copy upstream files into one output folder."""
    input_files = inputs.get('files', [])
    kwargs = input_files if isinstance(input_files, list) else [input_files]

    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    for kwarg in kwargs:
        origin_file = Path(get_tool_output(kwarg.source_uid, kwarg.source_k)[0])
        if origin_file.exists():
            new_file = output_folder / origin_file.name
            if new_file.exists() or new_file.is_symlink():
                new_file.unlink()
            shutil.copy2(origin_file, new_file)
        else:
            logger.warning(f'file {origin_file} did not exist!')

    return ToolOutput(result={'folder_path': output_folder})


def _normalize_mark_name(mark_name: str) -> str:
    normalized = mark_name.strip()
    if not normalized:
        msg = 'mark_name must not be empty.'
        raise ValueError(msg)
    if len(normalized) > PROJECT_FILE_KEYWORD_MAX_LENGTH:
        msg = f'mark_name must be no longer than {PROJECT_FILE_KEYWORD_MAX_LENGTH} characters.'
        raise ValueError(msg)
    if MARK_NAME_PATTERN.fullmatch(normalized) is None:
        msg = 'mark_name must start with a lowercase letter and contain only lowercase letters, digits, or _.'
        raise ValueError(msg)
    return normalized


def _register_mark(
    *,
    inputs: dict[str, TaskInput | list[TaskInput]],
    run_uid: str,
    mark_name: str,
    description: str,
    sample_scope: bool,
) -> ToolOutput:
    input_file = inputs.get('file')
    if input_file is None or isinstance(input_file, list):
        msg = 'A mark node requires exactly one file input.'
        raise ValueError(msg)

    normalized_name = _normalize_mark_name(mark_name)
    file_path, sample_name = get_tool_output(input_file.source_uid, input_file.source_k)
    marked_path = Path(file_path).absolute()
    if not output_path_exists(marked_path):
        raise FileNotFoundError(marked_path)

    with Session(engine) as session:
        parsed_run_uid = uuid.UUID(run_uid)
        run = session.get(RunInstance, parsed_run_uid)
        if run is None or run.project_id is None or run.workflow_uid is None:
            msg = 'Dynamic marks require a project workflow run.'
            raise ValueError(msg)

        if sample_scope:
            if run.sample_uid is None:
                msg = 'sample_mark requires a sample-level workflow run.'
                raise ValueError(msg)
            sample_service.upsert_dynamic_file(
                session,
                sample_uid=run.sample_uid,
                workflow_uid=run.workflow_uid,
                run_uid=run.uid,
                tag=normalized_name,
                file_path=str(marked_path),
                description=description,
            )
        else:
            if run.sample_uid is not None:
                msg = 'project_mark requires a project-level workflow run.'
                raise ValueError(msg)
            project_file_mapping_service.upsert_dynamic_mapping(
                session,
                project_id=run.project_id,
                workflow_uid=run.workflow_uid,
                run_uid=run.uid,
                keyword=normalized_name,
                file_path=str(marked_path),
                description=description,
            )

    return ToolOutput(result={'file_path': marked_path}, sample_name=sample_name)


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def project_mark_task(
    self: Task,  # noqa: ARG001
    result_dict: ToolOutput | None = None,  # noqa: ARG001
    *,
    inputs: dict[str, TaskInput | list[TaskInput]],
    run_uid: str,
    mark_name: str,
    description: str = '',
) -> ToolOutput:
    """Register an upstream path as a dynamic project mapping."""
    return _register_mark(
        inputs=inputs,
        run_uid=run_uid,
        mark_name=mark_name,
        description=description,
        sample_scope=False,
    )


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def sample_mark_task(
    self: Task,  # noqa: ARG001
    result_dict: ToolOutput | None = None,  # noqa: ARG001
    *,
    inputs: dict[str, TaskInput | list[TaskInput]],
    run_uid: str,
    mark_name: str,
    description: str = '',
) -> ToolOutput:
    """Register an upstream path as a dynamic sample-file mapping."""
    return _register_mark(
        inputs=inputs,
        run_uid=run_uid,
        mark_name=mark_name,
        description=description,
        sample_scope=True,
    )


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def collect_sample_marks_task(
    self: Task,  # noqa: ARG001
    result_dict: ToolOutput | None = None,  # noqa: ARG001
    *,
    project_id: int | None,
    sample_uid: str | None,
    mark_names: list[str],
    require_all_samples: bool,
    output_dir: str,
) -> ToolOutput:
    """Freeze current dynamic sample marks into a manifest and virtual mount collection."""
    if project_id is None:
        msg = 'Sample mark collections require a project-level workflow run.'
        raise ValueError(msg)
    if sample_uid is not None:
        msg = 'Sample mark collections cannot run inside a sample-level workflow.'
        raise ValueError(msg)

    normalized_names = list(dict.fromkeys(_normalize_mark_name(name) for name in mark_names))
    if not normalized_names:
        msg = 'mark_names must contain at least one mark.'
        raise ValueError(msg)

    output_folder = Path(output_dir).absolute()
    output_folder.mkdir(parents=True, exist_ok=True)
    mount_root = output_folder / 'mount_root'
    mount_root.mkdir(parents=True, exist_ok=True)

    with Session(engine) as session:
        sample_files = sample_crud.get_dynamic_files_by_tags(session, project_id, normalized_names)
        samples = list(
            session.scalars(select(Sample).where(Sample.project_id == project_id).order_by(Sample.sample_name))
        )

    found = {(str(sample_file.sample_uid), sample_file.tag) for sample_file, _ in sample_files}
    missing = [
        {'sample_uid': str(sample.uid), 'sample_name': sample.sample_name, 'mark_name': mark_name}
        for sample in samples
        for mark_name in normalized_names
        if (str(sample.uid), mark_name) not in found
    ]
    if require_all_samples and missing:
        summary = ', '.join(f'{item["sample_name"]}:{item["mark_name"]}' for item in missing)
        msg = f'Missing dynamic sample marks: {summary}'
        raise FileNotFoundError(msg)
    if not sample_files:
        msg = f'No dynamic sample marks found for: {", ".join(normalized_names)}'
        raise FileNotFoundError(msg)

    entries = []
    seen_targets: set[str] = set()
    for sample_file, sample_name in sample_files:
        source_path = Path(sample_file.file_path).absolute()
        if not output_path_exists(source_path):
            raise FileNotFoundError(source_path)

        sample_segment = to_safe_path_segment(sample_name, fallback=str(sample_file.sample_uid))
        mark_segment = to_safe_path_segment(sample_file.tag, fallback='mark')
        relative_path = f'{sample_segment}/{mark_segment}'
        if source_path.is_file():
            relative_path = f'{relative_path}/{to_safe_path_segment(source_path.name, fallback="file")}'
        if relative_path in seen_targets:
            msg = f'Duplicate collection target: {relative_path}'
            raise ValueError(msg)
        seen_targets.add(relative_path)

        entries.append(
            {
                'mount_name': relative_path,
                'source_path': str(source_path),
                'source_uid': str(sample_file.uid),
                'source_key': sample_file.tag,
                'sample_uid': str(sample_file.sample_uid),
                'sample_name': sample_name,
                'mark_name': sample_file.tag,
                'description': sample_file.description,
                'workflow_uid': (
                    str(sample_file.producer_workflow_uid) if sample_file.producer_workflow_uid is not None else None
                ),
                'run_uid': str(sample_file.producer_run_uid) if sample_file.producer_run_uid is not None else None,
                'kind': 'directory' if source_path.is_dir() else 'file',
            }
        )

    manifest_path = output_folder / 'collection_manifest.json'
    manifest = {
        'version': 1,
        'type': 'artifact_collection',
        'project_id': project_id,
        'mark_names': normalized_names,
        'require_all_samples': require_all_samples,
        'root_dir': str(mount_root),
        'entries': entries,
        'missing': missing,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

    return ToolOutput(
        result={
            'mount_collection': f'mountlist:{manifest_path}',
            'collection_manifest': manifest_path,
        }
    )


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def collect_mount_dirs_task(
    self: Task,  # noqa: ARG001
    result_dict: ToolOutput | None = None,  # noqa: ARG001
    *,
    inputs: dict[str, TaskInput | list[TaskInput]],
    output_dir: str,
) -> ToolOutput:
    """Collect multiple upstream directories into a mount-list manifest.

    The returned value is intentionally not a normal path. Downstream tool
    execution can recognize the ``mountlist:`` prefix and expand the manifest
    into multiple container mounts under one logical input directory.
    """
    input_dirs = inputs.get('dirs', [])
    kwargs = input_dirs if isinstance(input_dirs, list) else [input_dirs]
    if not kwargs:
        err_msg = 'collect_mount_dirs requires at least one input directory'
        raise ValueError(err_msg)

    output_folder = Path(output_dir).absolute()
    output_folder.mkdir(parents=True, exist_ok=True)
    mount_root = output_folder / 'mount_root'
    mount_root.mkdir(parents=True, exist_ok=True)

    entries = []
    sample_names = []
    for index, kwarg in enumerate(kwargs, start=1):
        source_path_str, sample_name = get_tool_output(kwarg.source_uid, kwarg.source_k)
        source_path = Path(source_path_str).absolute()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if not source_path.is_dir():
            raise NotADirectoryError(source_path)

        source_name = to_safe_path_segment(source_path.name, fallback='dir')
        mount_name = f'{index:03d}_{source_name}'
        entries.append(
            {
                'mount_name': mount_name,
                'source_path': str(source_path),
                'source_uid': kwarg.source_uid,
                'source_key': kwarg.source_k,
                'sample_name': sample_name,
            }
        )
        sample_names.append(sample_name)

    manifest_path = output_folder / 'mount_manifest.json'
    manifest = {
        'version': 1,
        'type': 'mount_dir_list',
        'root_dir': str(mount_root),
        'entries': entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    common_sample_name = get_prefix(sample_names)
    return ToolOutput(result={'mount_dir_list': f'mountlist:{manifest_path}'}, sample_name=common_sample_name)


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def bind_param_task(
    self: Task,  # noqa: ARG001
    result_dict: ToolOutput | None = None,  # noqa: ARG001
    *,
    inputs: dict[str, TaskInput],
    parameter: str,
) -> ToolOutput:
    """将文件绑定到指定参数名, 输出为 JSON 字符串."""
    kwarg = inputs.get('file')
    if kwarg is None:
        raise ValueError
    file_path, sample_name = get_tool_output(kwarg.source_uid, kwarg.source_k)
    return ToolOutput(result={'file_with_param': json.dumps({parameter: file_path})}, sample_name=sample_name)
