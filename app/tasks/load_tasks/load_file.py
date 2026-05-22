from celery import Task
from sqlmodel import Session

from app.core.session import engine
from app.models.workflow import ToolOutput
from app.tasks.base.base_task import BaseBioTask, get_prefix, resolve_path
from script.worker import celery_app


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
        file_path, file_tag = resolve_path(file, session, project_id, sample_uid)

    if not file_path.exists():
        err_msg = f'File "{file_path}" not found.'
        raise FileNotFoundError(err_msg)

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
        r1_path, r1_tag = resolve_path(r1, session, project_id, sample_uid)
        r2_path, r2_tag = resolve_path(r2, session, project_id, sample_uid)

    if not r1_path.exists():
        err_msg = f'File "{r1}" not found.'
        raise FileNotFoundError(err_msg)

    if not r2_path.exists():
        err_msg = f'File "{r2}" not found.'
        raise FileNotFoundError(err_msg)

    if sample_name is None:
        return ToolOutput(result={'r1': r1_path, 'r2': r2_path})

    common_tag = get_prefix([r1_tag, r2_tag])
    source_sample_name = f'{sample_name}_{common_tag}' if common_tag else sample_name
    return ToolOutput(result={'r1': r1_path, 'r2': r2_path}, sample_name=source_sample_name)
