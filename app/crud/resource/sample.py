import uuid
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import polars as pl
from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy import select as sa_select
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, select
from starlette.status import HTTP_404_NOT_FOUND

from app.models.project import Project
from app.models.resource import Sample, SampleFiles, SampleFileType
from app.schemas.resource import (
    SampleBatchImportResult,
    SampleGroupSummary,
    SampleListItem,
    SampleTagInfo,
    SampleUpdate,
)
from app.utils.file.hash import calculate_file_md5
from app.utils.system import get_file_size

_MIN_COMPOUND_EXTENSION_COUNT = 2


def _update_project_time(session: Session, project_id: int) -> None:
    """更新项目的 update_time.

    Args:
        session: 数据库会话
        project_id: 项目 ID
    """
    project_stmt = select(Project).where(Project.id == project_id)
    project = session.exec(project_stmt).one_or_none()

    if project:
        project.update_time = datetime.now(UTC)
        session.add(project)


def _calculate_file_info(file_path: str) -> tuple[str, int, str]:
    """计算文件信息.

    Args:
        file_path: 文件路径

    Returns:
        (file_format, file_size, md5_checksum) 元组

    Examples:
        >>> _calculate_file_info('/data/sample.fq.gz')
        ('fq.gz', 1024000, 'abc123...')
        >>> _calculate_file_info('/data/sample.tar.bz2')
        ('tar.bz2', 2048000, 'def456...')
    """
    path = Path(file_path)

    # 提取文件格式, 处理压缩文件 (.gz, .bz2, .xz 等)
    compression_suffixes = {'.gz', '.bz2', '.xz', '.bz', '.zip'}

    if path.suffix.lower() in compression_suffixes and len(path.suffixes) >= _MIN_COMPOUND_EXTENSION_COUNT:
        # 对于压缩文件, 取最后两个后缀, 如 .fq.gz, .tar.bz2
        file_format = ''.join(path.suffixes[-2:]).lstrip('.')
    elif path.suffix:
        # 普通文件, 直接取后缀
        file_format = path.suffix.lstrip('.')
    else:
        # 无后缀
        file_format = 'unknown'

    # 计算文件大小
    file_size = get_file_size(file_path)

    # 计算 MD5
    md5_checksum = calculate_file_md5(file_path)

    return file_format, file_size, md5_checksum


def _assign_default_tag(sample_file: SampleFiles) -> None:
    """为样本文件分配默认标签 (如果未指定).

    Args:
        sample_file: 样本文件对象
    """
    if sample_file.tag is None:
        # 根据 data_type 分配默认标签
        default_tags = {
            SampleFileType.SEQUENCING_R1: 'r1',
            SampleFileType.SEQUENCING_R2: 'r2',
            SampleFileType.SEQUENCING: 'single',
            SampleFileType.SPECTRUM: 'spectrum',
            SampleFileType.IMAGE: 'image',
        }
        # pyrefly: ignore [bad-assignment]
        sample_file.tag = default_tags.get(sample_file.data_type, 'unknown')


def insert(session: Session, sample: Sample) -> Sample:
    """创建新样本, 自动计算文件信息, 并更新项目时间."""
    session.add(sample)
    session.commit()
    session.refresh(sample)

    # 更新项目的 update_time
    _update_project_time(session, sample.project_id)
    session.commit()

    return sample


def batch_insert(session: Session, samples: list[Sample]) -> list[Sample]:
    """批量创建样本, 并在完成后统一更新项目时间.

    假定所有 sample 属于同一个 project.
    """
    project_ids = {s.project_id for s in samples}
    if len(project_ids) > 1:
        err_msg = f'All samples must belong to the same project, got project_ids: {project_ids}'
        raise ValueError(err_msg)

    session.add_all(samples)
    session.flush()
    for sample in samples:
        session.refresh(sample)

    _update_project_time(session, samples[0].project_id)
    session.commit()

    return samples


def delete(session: Session, sample_uid: uuid.UUID) -> None:
    """删除样本及其关联的文件, 并更新项目时间."""
    sample = get(session, sample_uid, with_files=True)
    if not sample:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found')

    project_id = sample.project_id

    session.delete(sample)
    _update_project_time(session, project_id)

    session.commit()


def update(session: Session, sample_uid: uuid.UUID, sample_update: SampleUpdate) -> Sample:
    """更新样本信息, 并更新项目时间."""
    sample = get(session, sample_uid)
    if not sample:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found')

    update_data = sample_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(sample, key, value)

    session.add(sample)

    # 更新项目的 update_time
    _update_project_time(session, sample.project_id)
    session.commit()
    session.refresh(sample)

    return sample


def get(session: Session, sample_uid: uuid.UUID, *, with_files: bool = False) -> Sample | None:
    """根据 UID 获取样本.

    Args:
        session: 数据库会话
        sample_uid: 样本 UID
        with_files: 是否预加载关联的文件 (使用 Relationship 优化查询)
    """
    stmt = select(Sample).where(Sample.uid == sample_uid)
    if with_files:
        stmt = stmt.options(selectinload(Sample.sample_files))  # type: ignore
    return session.exec(stmt).one_or_none()


def get_with_files(session: Session, sample_uid: uuid.UUID) -> Sample | None:
    """获取样本及其关联文件."""
    return get(session, sample_uid, with_files=True)


def get_page_by_user(
    session: Session, user_id: int, limit: int = 20, offset: int = 0
) -> tuple[list[SampleListItem], int]:
    """获取用户的所有样本."""
    stmt = (
        sa_select(
            col(Sample.uid),
            col(Sample.project_id),
            col(Sample.sample_name),
            col(Sample.meta_data),
            col(Sample.create_time),
            func.count(col(SampleFiles.uid)).label('file_count'),
        )
        .outerjoin(SampleFiles, col(Sample.uid) == SampleFiles.sample_uid)
        .where(col(Sample.owner_id) == user_id)
        .group_by(col(Sample.uid))
        .order_by(col(Sample.create_time).desc())
        .offset(offset)
        .limit(limit)
    )

    # pyrefly: ignore [no-matching-overload]
    results = session.exec(stmt).all()

    count_stmt = select(func.count()).select_from(Sample).where(Sample.owner_id == user_id)
    return [
        SampleListItem(
            uid=r.uid,
            project_id=r.project_id,
            sample_name=r.sample_name,
            meta_data=r.meta_data,
            create_time=r.create_time,
            file_count=r.file_count or 0,
        )
        for r in results
    ], session.exec(count_stmt).one()


def get_count(session: Session, project_id: int) -> int:
    """Return project sample counts."""
    stmt = select(func.count()).select_from(Sample).where(Sample.project_id == project_id)
    return session.exec(stmt).one()


def get_page_by_project(
    session: Session,
    project_id: int,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[SampleListItem], int]:
    """Return a page of a project's samples together with the total count."""
    stmt = (
        sa_select(
            col(Sample.uid),
            col(Sample.project_id),
            col(Sample.sample_name),
            col(Sample.meta_data),
            col(Sample.create_time),
            func.count(col(SampleFiles.uid)).label('file_count'),
        )
        .outerjoin(SampleFiles, col(Sample.uid) == SampleFiles.sample_uid)
        .where(col(Sample.project_id) == project_id)
        .group_by(col(Sample.uid))
        .order_by(col(Sample.create_time).desc())
        .offset(offset)
        .limit(limit)
    )

    # pyrefly: ignore [no-matching-overload]
    results = session.exec(stmt).all()

    return [
        SampleListItem(
            uid=r.uid,
            project_id=r.project_id,
            sample_name=r.sample_name,
            meta_data=r.meta_data,
            create_time=r.create_time,
            file_count=r.file_count or 0,
        )
        for r in results
    ], get_count(session, project_id)


def summarize_samples_by_project(session: Session, project_id: int) -> list[SampleGroupSummary]:
    """按 (meta_data, tag 集合) 聚合项目下样本, 返回组级摘要.

    用于大规模项目 (数百样本) 的概览, 避免逐样本返回造成 token 浪费.
    """
    stmt = (
        sa_select(
            col(Sample.uid),
            col(Sample.sample_name),
            col(Sample.meta_data),
            col(SampleFiles.tag),
            col(SampleFiles.data_type),
        )
        .outerjoin(SampleFiles, col(Sample.uid) == SampleFiles.sample_uid)
        .where(col(Sample.project_id) == project_id)
        .order_by(col(Sample.create_time))
    )
    # pyrefly: ignore [no-matching-overload]
    rows = session.exec(stmt).all()

    per_sample: dict[uuid.UUID, dict] = {}
    for r in rows:
        entry = per_sample.setdefault(
            r.uid,
            {'name': r.sample_name, 'meta': r.meta_data or {}, 'tags': {}},
        )
        if r.tag is not None:
            entry['tags'][r.tag] = r.data_type

    groups: dict[tuple, dict] = {}
    for entry in per_sample.values():
        tags_key = tuple(sorted(entry['tags'].items()))
        meta_key = tuple(sorted(entry['meta'].items()))
        key = (meta_key, tags_key)
        bucket = groups.setdefault(
            key,
            {'count': 0, 'meta': entry['meta'], 'tags': entry['tags'], 'names': []},
        )
        bucket['count'] += 1
        if len(bucket['names']) < 3:  # noqa: PLR2004
            bucket['names'].append(entry['name'])

    return [
        SampleGroupSummary(
            count=b['count'],
            meta_data=b['meta'],
            tags=[SampleTagInfo(tag=t, data_type=dt) for t, dt in sorted(b['tags'].items())],
            sample_name_examples=b['names'],
        )
        for b in groups.values()
    ]


def add_file(session: Session, sample_file: SampleFiles) -> SampleFiles:
    """为样本添加文件, 自动计算文件信息, 并更新项目时间."""
    # 验证样本是否存在
    sample = get(session, sample_file.sample_uid)
    if not sample:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Sample not found')

    # 分配默认标签
    _assign_default_tag(sample_file)
    # 自动计算文件信息
    file_format, file_size, md5_checksum = _calculate_file_info(sample_file.file_path)
    sample_file.file_format = file_format
    sample_file.file_size = file_size
    sample_file.md5_checksum = md5_checksum

    session.add(sample_file)

    # 更新项目的 update_time
    _update_project_time(session, sample.project_id)
    session.commit()
    session.refresh(sample_file)

    return sample_file


def delete_file(session: Session, file_uid: uuid.UUID) -> None:
    """删除样本文件, 并更新项目时间."""
    stmt = select(SampleFiles).where(SampleFiles.uid == file_uid)
    file = session.exec(stmt).one_or_none()

    if not file:
        raise HTTPException(HTTP_404_NOT_FOUND, 'File not found')

    # 获取关联的样本以获取 project_id
    sample = get(session, file.sample_uid)
    if sample:
        project_id = sample.project_id
        session.delete(file)
        # 更新项目的 update_time
        _update_project_time(session, project_id)
        session.commit()
    else:
        # 样本不存在, 仍然删除文件
        session.delete(file)
        session.commit()


def delete_file_by_name_tag(session: Session, project_id: int, sample_name: str, tag: str) -> bool:
    """根据项目 + 样本名 + tag 删除样本文件. 返回是否实际删除."""
    stmt = (
        select(SampleFiles)
        .join(Sample, col(Sample.uid) == SampleFiles.sample_uid)
        .where(Sample.project_id == project_id, Sample.sample_name == sample_name, SampleFiles.tag == tag)
    )
    file = session.exec(stmt).one_or_none()
    if not file:
        return False

    session.delete(file)
    _update_project_time(session, project_id)
    session.commit()
    return True


def get_files(session: Session, sample_uid: uuid.UUID) -> list[SampleFiles]:
    """获取样本的所有文件.

    直接返回 SampleFiles 列表, 由 FastAPI 根据 response_model 自动序列化
    """
    sample = get(session, sample_uid, with_files=True)
    if not sample:
        return []
    return sample.sample_files


def batch_import_from_csv(
    session: Session, csv_content: bytes, project_id: int, owner_id: int
) -> SampleBatchImportResult:
    """从 CSV 文件批量导入样本.

    Args:
        session: 数据库会话
        csv_content: CSV 文件内容 (bytes)
        project_id: 项目 ID
        owner_id: 所有者 ID

    Returns:
        SampleBatchImportResult: 导入结果
    """
    file_columns = {
        'r1': SampleFileType.SEQUENCING_R1,
        'r2': SampleFileType.SEQUENCING_R2,
        'single': SampleFileType.SEQUENCING,
        'spectrum': SampleFileType.SPECTRUM,
        'image': SampleFileType.IMAGE,
    }

    result = SampleBatchImportResult(success=0, failed=0, errors=[], samples=[])

    try:
        df = pl.read_csv(BytesIO(csv_content), encoding='utf-8')
    except pl.exceptions.ComputeError as e:
        raise HTTPException(400, f'CSV 文件格式错误: {e!s}') from e

    if 'sample_name' not in df.columns:
        raise HTTPException(400, 'CSV 文件缺少必需列: sample_name')

    fixed_columns = {'sample_name', 'r1', 'r2', 'single', 'spectrum', 'image'}
    meta_columns = [col for col in df.columns if col not in fixed_columns]

    for row_idx, row in enumerate(df.iter_rows(named=True), start=1):
        try:
            sample_name = row.get('sample_name')

            if not sample_name or not str(sample_name).strip():
                result.failed += 1
                result.errors.append(
                    {
                        'row': row_idx,
                        'sample_name': sample_name or '<空>',
                        'error': '样本名称不能为空',
                    }
                )
                continue

            sample_name = str(sample_name).strip()

            meta_data = {
                col: str(value) for col in meta_columns if (value := row.get(col)) is not None and str(value).strip()
            }

            sample = Sample(
                owner_id=owner_id,
                project_id=project_id,
                sample_name=sample_name,
                meta_data=meta_data,
            )
            session.add(sample)
            session.flush()

            for col_name, file_type in file_columns.items():
                file_path = row.get(col_name)
                if file_path and str(file_path).strip():
                    file_path = str(file_path).strip()
                    file_format, file_size, md5_checksum = _calculate_file_info(file_path)
                    sample_file = SampleFiles(
                        sample_uid=sample.uid,
                        data_type=file_type,
                        file_path=file_path,
                        file_format=file_format,
                        file_size=file_size,
                        md5_checksum=md5_checksum,
                    )
                    session.add(sample_file)

            session.commit()
            result.success += 1
            result.samples.append(sample.uid)

        except Exception as e:
            session.rollback()
            result.failed += 1
            result.errors.append(
                {
                    'row': row_idx,
                    'sample_name': row.get('sample_name', '<未知>'),
                    'error': str(e),
                }
            )

    if result.success > 0:
        _update_project_time(session, project_id)
        session.commit()

    return result
