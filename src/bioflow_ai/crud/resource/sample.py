import uuid

from sqlalchemy import and_, func, select
from sqlalchemy import delete as sa_delete
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.sql import Select

from bioflow_ai.models.project import Project
from bioflow_ai.models.resource import Sample, SampleFiles
from bioflow_ai.models.workflow import Workflow
from bioflow_ai.schemas.resource import SampleGroupSummary, SampleListItem, SampleUpdate
from bioflow_ai.utils.identifiers import UUIDLike, as_uuid


def insert(session: Session, sample: Sample) -> Sample:
    """Stage a new sample."""
    session.add(sample)
    session.flush()
    session.refresh(sample)
    return sample


def batch_insert(session: Session, samples: list[Sample]) -> list[Sample]:
    """Stage a batch of samples."""
    session.add_all(samples)
    session.flush()
    for sample in samples:
        session.refresh(sample)
    return samples


def delete(session: Session, sample_uid: UUIDLike) -> Sample | None:
    """Stage deletion of a sample and return the deleted record."""
    sample = get(session, sample_uid, with_files=True)
    if sample is None:
        return None
    session.delete(sample)
    session.flush()
    return sample


def update(session: Session, sample_uid: UUIDLike, data: SampleUpdate) -> Sample | None:
    """Stage editable sample fields."""
    sample = get(session, sample_uid)
    if sample is None:
        return None
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(sample, field, value)
    session.flush()
    session.refresh(sample)
    return sample


def get(session: Session, sample_uid: UUIDLike, *, with_files: bool = False) -> Sample | None:
    """Return a sample by UID, optionally eager-loading files."""
    statement = select(Sample).where(Sample.uid == as_uuid(sample_uid))
    if with_files:
        statement = statement.options(selectinload(Sample.sample_files))
    return session.scalar(statement)


def get_by_uids(
    session: Session,
    sample_uids: list[uuid.UUID],
    *,
    project_id: int | None = None,
) -> list[Sample]:
    """Return samples matching the given UIDs in caller-supplied order."""
    if not sample_uids:
        return []
    statement = select(Sample).where(Sample.uid.in_(sample_uids))
    if project_id is not None:
        statement = statement.where(Sample.project_id == project_id)
    samples = session.scalars(statement)
    sample_by_uid = {sample.uid: sample for sample in samples}
    return [sample_by_uid[uid] for uid in sample_uids if uid in sample_by_uid]


def get_with_files(session: Session, sample_uid: UUIDLike) -> Sample | None:
    """Return a sample with its file collection eagerly loaded."""
    return get(session, sample_uid, with_files=True)


def _sample_list_statement() -> Select:
    return select(
        Sample.uid,
        Sample.project_id,
        Sample.sample_name,
        Sample.meta_data,
        Sample.create_time,
        func.count(SampleFiles.uid).label('file_count'),
    ).outerjoin(SampleFiles, Sample.uid == SampleFiles.sample_uid)


def _list_items(session: Session, statement: Select) -> list[SampleListItem]:
    return [
        SampleListItem(
            uid=row.uid,
            project_id=row.project_id,
            sample_name=row.sample_name,
            meta_data=row.meta_data,
            create_time=row.create_time,
            file_count=row.file_count or 0,
        )
        for row in session.execute(statement)
    ]


def get_page_by_user(
    session: Session,
    user_id: int | None,
    limit: int = 20,
    offset: int = 0,
    *,
    public_only: bool = False,
) -> tuple[list[SampleListItem], int]:
    """Return a page of samples owned by a user."""
    statement = _sample_list_statement()
    count_statement = select(func.count()).select_from(Sample)
    if public_only:
        statement = statement.join(Project, Sample.project_id == Project.id).where(Project.public.is_(True))
        count_statement = count_statement.join(Project, Sample.project_id == Project.id).where(Project.public.is_(True))
    if user_id is not None:
        statement = statement.where(Sample.owner_id == user_id)
        count_statement = count_statement.where(Sample.owner_id == user_id)
    statement = statement.group_by(Sample.uid).order_by(Sample.create_time.desc()).offset(offset).limit(limit)
    return _list_items(session, statement), session.scalar(count_statement) or 0


def get_count(session: Session, project_id: int) -> int:
    """Return a project's sample count."""
    statement = select(func.count()).select_from(Sample).where(Sample.project_id == project_id)
    return session.scalar(statement) or 0


def get_page_by_project(
    session: Session,
    project_id: int,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[SampleListItem], int]:
    """Return a page of a project's samples together with the total count."""
    statement = (
        _sample_list_statement()
        .where(Sample.project_id == project_id)
        .group_by(Sample.uid)
        .order_by(Sample.create_time.desc())
        .offset(offset)
        .limit(limit)
    )
    return _list_items(session, statement), get_count(session, project_id)


def summarize_samples_by_project(session: Session, project_id: int) -> list[SampleGroupSummary]:
    """Group project samples by metadata and non-dynamic file-tag sets."""
    statement = (
        select(Sample.uid, Sample.sample_name, Sample.meta_data, SampleFiles.tag)
        .outerjoin(
            SampleFiles,
            and_(
                Sample.uid == SampleFiles.sample_uid,
                SampleFiles.is_dynamic.is_(False),
            ),
        )
        .where(Sample.project_id == project_id)
        .order_by(Sample.create_time)
    )
    per_sample: dict[uuid.UUID, dict] = {}
    for row in session.execute(statement):
        entry = per_sample.setdefault(
            row.uid,
            {'name': row.sample_name, 'meta': row.meta_data or {}, 'tags': set()},
        )
        if row.tag is not None:
            entry['tags'].add(row.tag)

    groups: dict[tuple, dict] = {}
    for entry in per_sample.values():
        key = (tuple(sorted(entry['meta'].items())), tuple(sorted(entry['tags'])))
        bucket = groups.setdefault(
            key,
            {'count': 0, 'meta': entry['meta'], 'tags': entry['tags'], 'names': []},
        )
        bucket['count'] += 1
        if len(bucket['names']) < 3:  # noqa: PLR2004
            bucket['names'].append(entry['name'])

    return [
        SampleGroupSummary(
            count=bucket['count'],
            meta_data=bucket['meta'],
            tags=sorted(bucket['tags']),
            sample_name_examples=bucket['names'],
        )
        for bucket in groups.values()
    ]


def insert_file(session: Session, sample_file: SampleFiles) -> SampleFiles:
    """Stage a new sample file."""
    session.add(sample_file)
    session.flush()
    session.refresh(sample_file)
    return sample_file


def get_file(session: Session, file_uid: UUIDLike) -> SampleFiles | None:
    """Return a sample file by UID."""
    return session.get(SampleFiles, as_uuid(file_uid))


def get_file_by_tag(session: Session, sample_uid: UUIDLike, tag: str) -> SampleFiles | None:
    """Return the file occupying a tag within a sample."""
    statement = select(SampleFiles).where(
        SampleFiles.sample_uid == as_uuid(sample_uid),
        SampleFiles.tag == tag,
    )
    return session.scalar(statement)


def delete_dynamic_files_for_workflow(
    session: Session,
    *,
    sample_uid: UUIDLike,
    workflow_uid: uuid.UUID,
) -> None:
    """Stage deletion of dynamic sample files owned by a workflow."""
    session.execute(
        sa_delete(SampleFiles).where(
            SampleFiles.sample_uid == as_uuid(sample_uid),
            SampleFiles.is_dynamic.is_(True),
            SampleFiles.producer_workflow_uid == workflow_uid,
        )
    )


def delete_dynamic_files_by_producer(
    session: Session, workflow_uid: uuid.UUID, *, project_id: int | None = None
) -> None:
    """Stage deletion of dynamic sample files registered by a workflow, optionally within a project."""
    statement = sa_delete(SampleFiles).where(
        SampleFiles.is_dynamic.is_(True),
        SampleFiles.producer_workflow_uid == workflow_uid,
    )
    if project_id is not None:
        statement = statement.where(
            SampleFiles.sample_uid.in_(select(Sample.uid).where(Sample.project_id == project_id))
        )
    session.execute(statement)


def get_dynamic_files_owned_elsewhere(
    session: Session,
    project_id: int,
    tag: str,
    workflow_uid: uuid.UUID | None,
    *,
    limit: int = 3,
) -> list[tuple[SampleFiles, str]]:
    """Return a few dynamic files using a tag whose producer is another existing workflow."""
    statement = (
        select(SampleFiles, Sample.sample_name)
        .join(Sample, Sample.uid == SampleFiles.sample_uid)
        .where(
            Sample.project_id == project_id,
            SampleFiles.tag == tag,
            SampleFiles.is_dynamic.is_(True),
            SampleFiles.producer_workflow_uid.in_(select(Workflow.uid)),
        )
        .order_by(Sample.sample_name)
    )
    if workflow_uid is not None:
        statement = statement.where(SampleFiles.producer_workflow_uid != workflow_uid)
    return [(sample_file, sample_name) for sample_file, sample_name in session.execute(statement.limit(limit))]


def get_dynamic_files_by_tags(
    session: Session,
    project_id: int,
    tags: list[str],
) -> list[tuple[SampleFiles, str]]:
    """Return dynamic sample files and their sample names for collection."""
    statement = (
        select(SampleFiles, Sample.sample_name)
        .join(Sample, Sample.uid == SampleFiles.sample_uid)
        .where(
            Sample.project_id == project_id,
            SampleFiles.is_dynamic.is_(True),
            SampleFiles.tag.in_(tags),
        )
        .order_by(Sample.sample_name, SampleFiles.tag)
    )
    return [(sample_file, sample_name) for sample_file, sample_name in session.execute(statement)]


def get_files_by_project(
    session: Session,
    project_id: int,
) -> list[tuple[SampleFiles, str]]:
    """Return all sample-file markers and sample names in a project."""
    statement = (
        select(SampleFiles, Sample.sample_name)
        .join(Sample, Sample.uid == SampleFiles.sample_uid)
        .where(Sample.project_id == project_id)
        .order_by(SampleFiles.is_dynamic, SampleFiles.tag, Sample.sample_name)
    )
    return [(sample_file, sample_name) for sample_file, sample_name in session.execute(statement)]


def get_files_by_tag_in_project(
    session: Session,
    project_id: int,
    tag: str,
    *,
    limit: int = 3,
    is_dynamic: bool | None = None,
) -> list[tuple[SampleFiles, str]]:
    """Return a few sample files occupying a tag, optionally filtered by dynamic status."""
    statement = (
        select(SampleFiles, Sample.sample_name)
        .join(Sample, Sample.uid == SampleFiles.sample_uid)
        .where(Sample.project_id == project_id, SampleFiles.tag == tag)
        .order_by(Sample.sample_name)
    )
    if is_dynamic is not None:
        statement = statement.where(SampleFiles.is_dynamic.is_(is_dynamic))
    return [(sample_file, sample_name) for sample_file, sample_name in session.execute(statement.limit(limit))]


def delete_file(session: Session, file_uid: UUIDLike) -> SampleFiles | None:
    """Stage deletion of a sample file and return the deleted record."""
    sample_file = get_file(session, file_uid)
    if sample_file is None:
        return None
    session.delete(sample_file)
    session.flush()
    return sample_file


def get_file_by_name_tag(
    session: Session,
    project_id: int,
    sample_name: str,
    tag: str,
) -> SampleFiles | None:
    """Return a sample file by project, sample name, and tag."""
    statement = (
        select(SampleFiles)
        .join(Sample, Sample.uid == SampleFiles.sample_uid)
        .where(
            Sample.project_id == project_id,
            Sample.sample_name == sample_name,
            SampleFiles.tag == tag,
        )
    )
    return session.scalar(statement)


def get_files(session: Session, sample_uid: UUIDLike) -> list[SampleFiles]:
    """Return all files belonging to a sample."""
    sample = get_with_files(session, sample_uid)
    return [] if sample is None else sample.sample_files
