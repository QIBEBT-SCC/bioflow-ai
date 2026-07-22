import re
from pathlib import Path

from celery import Task
from sqlmodel import Session

from app.core.session import engine
from app.crud import project_file_mapping as project_file_mapping_crud
from app.crud.resource import sample as sample_crud
from app.models.workflow import ToolOutput
from app.services.output_validation import output_path_exists
from app.tasks.base.base_task import BaseBioTask, get_prefix, resolve_path
from script.worker import celery_app

_FASTQ_PATTERN = re.compile('fastq', re.IGNORECASE)
_COMPRESSION_SUFFIXES = {'.gz', '.bz2', '.xz', '.bz', '.zip'}
_MIN_COMPOUND_EXTENSION_COUNT = 2


def _get_file_format(file_path: Path) -> str:
    """Return a display file format based on the file suffix."""
    if file_path.suffix.lower() in _COMPRESSION_SUFFIXES and len(file_path.suffixes) >= _MIN_COMPOUND_EXTENSION_COUNT:
        return ''.join(file_path.suffixes[-2:]).lstrip('.')

    if file_path.suffix:
        return file_path.suffix.lstrip('.')

    return 'unknown'


def _rename_fastq_file(file_path: Path) -> Path:
    """Rename fastq-containing file names to fq-containing names."""
    if not file_path.is_file():
        return file_path

    target_path = file_path.with_name(_FASTQ_PATTERN.sub('fq', file_path.name))
    if target_path == file_path:
        return file_path

    if target_path.exists():
        err_msg = f'Cannot rename "{file_path}" to "{target_path}": target file already exists.'
        raise FileExistsError(err_msg)

    file_path.rename(target_path)
    return target_path


def _update_loaded_path_reference(
    path_or_reference: str,
    normalized_path: Path,
    session: Session,
    project_id: int | None,
    sample_uid: str | None,
) -> None:
    """Persist a renamed file path for project/sample references."""
    if path_or_reference.startswith('proj:') and project_id is not None:
        keyword = path_or_reference.split(':', 1)[1]
        mapping = project_file_mapping_crud.get_by_keyword(session, project_id, keyword)
        if mapping is not None:
            mapping.file_path = str(normalized_path)
            session.add(mapping)
        return

    if path_or_reference.startswith('sample:') and sample_uid is not None:
        tag = path_or_reference.split(':', 1)[1]
        sample = sample_crud.get_with_files(session, sample_uid)  # type: ignore
        if sample is None:
            return

        matching_file = next((f for f in sample.sample_files if f.tag == tag), None)
        if matching_file is not None:
            matching_file.file_path = str(normalized_path)
            matching_file.file_format = _get_file_format(normalized_path)
            session.add(matching_file)


def _resolve_and_normalize_path(
    path_or_reference: str,
    session: Session,
    project_id: int | None,
    sample_uid: str | None,
) -> tuple[Path, str | None]:
    """Resolve a load path and normalize fastq file names to fq."""
    file_path, file_tag = resolve_path(path_or_reference, session, project_id, sample_uid)
    if not output_path_exists(file_path):
        err_msg = f'File "{file_path}" not found.'
        raise FileNotFoundError(err_msg)

    normalized_path = _rename_fastq_file(file_path)
    if normalized_path != file_path:
        _update_loaded_path_reference(path_or_reference, normalized_path, session, project_id, sample_uid)
        session.commit()

    return normalized_path, file_tag


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def load_file_task(
    self: Task,  # noqa: ARG001
    result_dict: ToolOutput | None = None,  # noqa: ARG001
    *,
    file: str,
    project_id: int | None = None,
    sample_uid: str | None = None,
    sample_name: str | None = None,
) -> ToolOutput:
    """加载普通文件."""
    with Session(engine) as session:
        file_path, file_tag = _resolve_and_normalize_path(file, session, project_id, sample_uid)

    if sample_name is None:
        return ToolOutput(result={'file_path': file_path})

    source_sample_name = f'{sample_name}_{file_tag}' if file_tag else sample_name
    return ToolOutput(result={'file_path': file_path}, sample_name=source_sample_name)


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def load_sequence_task(
    self: Task,  # noqa: ARG001
    result_dict: ToolOutput | None = None,  # noqa: ARG001
    *,
    r1: str,
    r2: str,
    project_id: int | None = None,
    sample_uid: str | None = None,
    sample_name: str | None = None,
) -> ToolOutput:
    """加载双端序列文件."""
    with Session(engine) as session:
        r1_path, r1_tag = _resolve_and_normalize_path(r1, session, project_id, sample_uid)
        r2_path, r2_tag = _resolve_and_normalize_path(r2, session, project_id, sample_uid)

    if sample_name is None:
        return ToolOutput(result={'r1': r1_path, 'r2': r2_path})

    common_tag = get_prefix([r1_tag, r2_tag])
    source_sample_name = f'{sample_name}_{common_tag}' if common_tag else sample_name
    return ToolOutput(result={'r1': r1_path, 'r2': r2_path}, sample_name=source_sample_name)
