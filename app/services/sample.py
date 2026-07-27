import uuid
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import polars as pl
from sqlalchemy.orm import Session

from app.crud.resource import sample as sample_crud
from app.models.project import Project
from app.models.resource import Sample, SampleFiles
from app.schemas.resource import SampleBatchImportResult, SampleUpdate
from app.services._transaction import committing
from app.utils.file.hash import calculate_file_md5
from app.utils.identifiers import UUIDLike
from app.utils.system import get_file_size

_MIN_COMPOUND_EXTENSION_COUNT = 2
_CSV_FILE_COLUMNS = {'r1', 'r2', 'single', 'spectrum', 'image'}


class SampleServiceError(ValueError):
    """Base error for sample application operations."""


class SampleNotFoundError(SampleServiceError):
    """Raised when a requested sample does not exist."""


class SampleFileNotFoundError(SampleServiceError):
    """Raised when a requested sample file does not exist."""


class SampleBatchProjectMismatchError(SampleServiceError):
    """Raised when one batch contains samples from multiple projects."""


class SampleCsvError(SampleServiceError):
    """Raised when a sample CSV cannot be parsed or lacks required columns."""


class DynamicSampleFileConflictError(SampleServiceError):
    """Raised when a dynamic mark conflicts with an existing sample file."""


def _update_project_time(session: Session, project_id: int) -> None:
    project = session.get(Project, project_id)
    if project is not None:
        project.update_time = datetime.now(UTC)


def _calculate_file_info(file_path: str) -> tuple[str, int, str]:
    path = Path(file_path)
    compression_suffixes = {'.gz', '.bz2', '.xz', '.bz', '.zip'}
    if path.suffix.lower() in compression_suffixes and len(path.suffixes) >= _MIN_COMPOUND_EXTENSION_COUNT:
        file_format = ''.join(path.suffixes[-2:]).lstrip('.')
    elif path.suffix:
        file_format = path.suffix.lstrip('.')
    else:
        file_format = 'unknown'
    return file_format, get_file_size(file_path), calculate_file_md5(file_path)


def create_sample(session: Session, sample: Sample) -> Sample:
    """Create a sample and update its project's modification time."""
    with committing(session):
        sample_crud.insert(session, sample)
        _update_project_time(session, sample.project_id)
    return sample


def create_samples(session: Session, samples: list[Sample]) -> list[Sample]:
    """Create one same-project sample batch and commit it atomically."""
    if not samples:
        return []
    project_ids = {sample.project_id for sample in samples}
    if len(project_ids) != 1:
        msg = f'All samples must belong to the same project, got project_ids: {project_ids}'
        raise SampleBatchProjectMismatchError(msg)
    with committing(session):
        sample_crud.batch_insert(session, samples)
        _update_project_time(session, samples[0].project_id)
    return samples


def update_sample(session: Session, sample_uid: UUIDLike, data: SampleUpdate) -> Sample:
    """Update a sample and its project's modification time."""
    sample = sample_crud.get(session, sample_uid)
    if sample is None:
        raise SampleNotFoundError
    with committing(session):
        updated = sample_crud.update(session, sample_uid, data)
        _update_project_time(session, sample.project_id)
    if updated is None:
        raise SampleNotFoundError
    return updated


def delete_sample(session: Session, sample_uid: UUIDLike) -> None:
    """Delete a sample and update its project's modification time."""
    sample = sample_crud.get(session, sample_uid, with_files=True)
    if sample is None:
        raise SampleNotFoundError
    project_id = sample.project_id
    with committing(session):
        sample_crud.delete(session, sample_uid)
        _update_project_time(session, project_id)


def add_file(session: Session, sample_file: SampleFiles) -> SampleFiles:
    """Calculate file metadata, attach the file, and commit."""
    sample = sample_crud.get(session, sample_file.sample_uid)
    if sample is None:
        raise SampleNotFoundError
    sample_file.file_format, sample_file.file_size, sample_file.md5_checksum = _calculate_file_info(
        sample_file.file_path
    )
    with committing(session):
        sample_crud.insert_file(session, sample_file)
        _update_project_time(session, sample.project_id)
    session.refresh(sample_file)
    return sample_file


def upsert_dynamic_file(
    session: Session,
    *,
    sample_uid: UUIDLike,
    workflow_uid: uuid.UUID,
    run_uid: uuid.UUID,
    tag: str,
    file_path: str,
    description: str,
) -> SampleFiles:
    """Create or replace one workflow-owned dynamic sample file."""
    sample = sample_crud.get(session, sample_uid)
    if sample is None:
        raise SampleNotFoundError
    sample_file = sample_crud.get_file_by_tag(session, sample_uid, tag)
    if sample_file is not None and not sample_file.is_dynamic:
        msg = f'Dynamic mark "{tag}" conflicts with initial sample resource "sample:{tag}".'
        raise DynamicSampleFileConflictError(msg)
    if sample_file is not None and sample_file.producer_workflow_uid != workflow_uid:
        msg = f'Dynamic sample mark "{tag}" is already owned by workflow {sample_file.producer_workflow_uid}.'
        raise DynamicSampleFileConflictError(msg)

    path = Path(file_path)
    file_format = 'directory' if path.is_dir() else ''.join(path.suffixes).lstrip('.') or 'unknown'
    if sample_file is None:
        sample_file = SampleFiles(
            sample_uid=sample.uid,
            file_path=file_path,
            file_format=file_format,
            tag=tag,
            description=description,
            is_dynamic=True,
            producer_workflow_uid=workflow_uid,
            producer_run_uid=run_uid,
        )
    else:
        sample_file.file_path = file_path
        sample_file.file_format = file_format
        sample_file.file_size = 0
        sample_file.md5_checksum = None
        sample_file.description = description
        sample_file.is_dynamic = True
        sample_file.producer_workflow_uid = workflow_uid
        sample_file.producer_run_uid = run_uid
        sample_file.uploaded_time = datetime.now(UTC)

    with committing(session):
        sample_crud.insert_file(session, sample_file)
        _update_project_time(session, sample.project_id)
    return sample_file


def delete_file(session: Session, file_uid: UUIDLike) -> None:
    """Delete a sample file and update its project's modification time."""
    sample_file = sample_crud.get_file(session, file_uid)
    if sample_file is None:
        raise SampleFileNotFoundError
    sample = sample_crud.get(session, sample_file.sample_uid)
    with committing(session):
        sample_crud.delete_file(session, file_uid)
        if sample is not None:
            _update_project_time(session, sample.project_id)


def delete_file_by_name_tag(session: Session, project_id: int, sample_name: str, tag: str) -> bool:
    """Delete a sample file by project, sample name, and tag."""
    sample_file = sample_crud.get_file_by_name_tag(session, project_id, sample_name, tag)
    if sample_file is None:
        return False
    with committing(session):
        sample_crud.delete_file(session, sample_file.uid)
        _update_project_time(session, project_id)
    return True


def import_samples_from_csv(
    session: Session,
    csv_content: bytes,
    project_id: int,
    owner_id: int,
) -> SampleBatchImportResult:
    """Import CSV rows with per-row transaction isolation."""
    result = SampleBatchImportResult(success=0, failed=0)
    try:
        frame = pl.read_csv(BytesIO(csv_content), encoding='utf-8')
    except pl.exceptions.ComputeError as error:
        msg = f'CSV 文件格式错误: {error!s}'
        raise SampleCsvError(msg) from error
    if 'sample_name' not in frame.columns:
        msg = 'CSV 文件缺少必需列: sample_name'
        raise SampleCsvError(msg)

    fixed_columns = {'sample_name', *_CSV_FILE_COLUMNS}
    meta_columns = [column for column in frame.columns if column not in fixed_columns]
    for row_index, row in enumerate(frame.iter_rows(named=True), start=1):
        try:
            sample_name_value = row.get('sample_name')
            if not sample_name_value or not str(sample_name_value).strip():
                result.failed += 1
                result.errors.append(
                    {'row': row_index, 'sample_name': sample_name_value or '<空>', 'error': '样本名称不能为空'}
                )
                continue
            sample_name = str(sample_name_value).strip()
            meta_data = {
                column: str(value)
                for column in meta_columns
                if (value := row.get(column)) is not None and str(value).strip()
            }
            sample = Sample(
                owner_id=owner_id,
                project_id=project_id,
                sample_name=sample_name,
                meta_data=meta_data,
            )
            with committing(session):
                sample_crud.insert(session, sample)
                for column_name in _CSV_FILE_COLUMNS:
                    file_value = row.get(column_name)
                    if file_value and str(file_value).strip():
                        file_path = str(file_value).strip()
                        file_format, file_size, md5_checksum = _calculate_file_info(file_path)
                        sample_crud.insert_file(
                            session,
                            SampleFiles(
                                sample_uid=sample.uid,
                                tag=column_name,
                                file_path=file_path,
                                file_format=file_format,
                                file_size=file_size,
                                md5_checksum=md5_checksum,
                            ),
                        )
            result.success += 1
            result.samples.append(sample.uid)
        except Exception as error:  # Preserve per-row import error reporting.
            result.failed += 1
            result.errors.append(
                {
                    'row': row_index,
                    'sample_name': row.get('sample_name', '<未知>'),
                    'error': str(error),
                }
            )

    if result.success:
        with committing(session):
            _update_project_time(session, project_id)
    return result
