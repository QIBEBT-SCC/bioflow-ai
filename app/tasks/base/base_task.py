import json
import platform
import socket
import time
from datetime import UTC, datetime
from itertools import takewhile
from pathlib import Path

import celery
from loguru import logger
from redis import Redis
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

import app.crud.run_instance as run_crud
import app.crud.task_instance as task_crud
from app.core.session import engine
from app.crud import project_file_mapping as project_file_mapping_crud
from app.crud.resource import sample as sample_crud
from app.models.tools import ToolDefine
from app.models.workflow import TaskInstance, TaskMonitor
from app.schemas.run_instance import RunUpdate
from app.schemas.task_instance import TaskUpdate
from app.schemas.tools import ToolTaskInfo
from app.schemas.workflow_config import Status, ToolOutput
from app.services import workflow_runtime
from app.utils.identifiers import as_uuid


class BaseBioTask(celery.Task):
    def before_start(self, task_id: str, *args, **kwargs) -> None:
        """任务开始前的钩子 - 落盘任务启动状态."""
        try:
            with Session(engine) as session:
                task = workflow_runtime.update_task(
                    session,
                    task_id,
                    TaskUpdate(
                        status=Status.RUNNING,
                        system=platform.platform(),
                        hostname=socket.gethostname(),
                        start_time=datetime.now(UTC),
                    ),
                )
                logger.info(f'Task {task.name}({task_id}) START')

                instance_uid = str(task.instance_uid)

            _mark_run_instance_running(instance_uid)
        except Exception as e:
            logger.exception(f'更新任务 {task_id} 启动状态失败: {e}')

    def on_success(self, retval, task_id, args, kwargs) -> None:  # noqa: ANN001
        """任务成功后的钩子 - 落盘任务终态并 flush 监控日志."""
        try:
            with Session(engine) as session:
                tool_output = retval if isinstance(retval, ToolOutput) else ToolOutput.model_validate(retval)

                task = workflow_runtime.update_task(
                    session,
                    task_id,
                    TaskUpdate(
                        status=Status.SUCCESS,
                        tool_output=tool_output,
                        end_time=datetime.now(UTC),
                    ),
                )
                logger.success(f'Task {task.name}({task_id}) SUCCESS')

                instance_uid = str(task.instance_uid)

            _mark_run_instance_success_if_complete(instance_uid)
        except Exception as e:
            logger.exception(f'更新任务 {task_id} 成功状态失败: {e}')

        try:
            _flush_monitor_to_db(task_id)
        except Exception as e:
            logger.warning(f'flush 监控日志失败: {e}')

    def on_failure(self, exc, task_id, args, kwargs, einfo) -> None:  # noqa: ANN001
        """任务失败后的钩子 - 落盘任务终态并 flush 监控日志."""
        failure_base_dir: str | None = None
        try:
            with Session(engine) as session:
                task = workflow_runtime.update_task(
                    session,
                    task_id,
                    TaskUpdate(status=Status.ERROR, end_time=datetime.now(UTC)),
                )
                logger.error(f'Task {task.name}({task_id}) ERROR')

                instance_uid = str(task.instance_uid)
                failure_base_dir = task.base_dir

            _mark_run_instance_error(instance_uid)
        except Exception as e:
            logger.error(f'更新任务 {task_id} 失败状态失败: {e}')

        if failure_base_dir:
            try:
                _append_failure_log(failure_base_dir, exc, einfo)
            except Exception as e:
                logger.warning(f'写入任务 {task_id} 失败日志失败: {e}')

        try:
            _flush_monitor_to_db(task_id)
        except Exception as e:
            logger.warning(f'flush 监控日志失败: {e}')


# ---------------------------------------------------------------------------
# Runtime persistence helpers
# ---------------------------------------------------------------------------

_TASK_NOT_FOUND = 'Task not found'


def _append_failure_log(base_dir: str, exc: object, einfo: object) -> None:
    """Persist a durable error snapshot beside task outputs for later diagnosis."""
    output_dir = Path(base_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / '11_run.log'
    prefix = '\n' if log_path.exists() and log_path.stat().st_size else ''
    traceback_text = getattr(einfo, 'traceback', None) or str(einfo)
    with log_path.open('a', encoding='utf-8') as log_file:
        log_file.write(f'{prefix}[celery failure]\n{type(exc).__name__}: {exc}\n{traceback_text}\n')


def _get_redis() -> Redis:
    # Delayed because script.worker imports the task modules during worker startup.
    from script.worker import monitor_redis_client

    return monitor_redis_client


def _require_task(session: Session, task_uid: str) -> TaskInstance:
    """Return a task instance or raise the runtime domain error."""
    task = task_crud.get(session, task_uid)
    if task is None:
        raise workflow_runtime.WorkflowRuntimeNotFoundError(_TASK_NOT_FOUND)
    return task


def _mark_run_instance_running(run_uid: str) -> None:
    """Mark a run instance as running when any task starts."""
    try:
        with Session(engine) as session:
            run = run_crud.get(session, run_uid)
            if run is None:
                return
            if run.status not in {Status.RUNNING, Status.SUCCESS, Status.ERROR}:
                workflow_runtime.update_run(
                    session,
                    run_uid,
                    RunUpdate(status=Status.RUNNING, start_time=datetime.now(tz=UTC)),
                )
    except Exception as e:
        logger.warning(f'更新 RunInstance RUNNING 状态失败: {e}')


def _mark_run_instance_success_if_complete(run_uid: str) -> None:
    """Mark a run instance as success when all tasks have succeeded."""
    try:
        with Session(engine) as session:
            stats = run_crud.get_execution_task_stats(session, run_uid)
            if 0 < stats.total == stats.success:
                workflow_runtime.update_run(
                    session,
                    run_uid,
                    RunUpdate(status=Status.SUCCESS, end_time=datetime.now(tz=UTC)),
                )
    except Exception as e:
        logger.warning(f'更新 RunInstance SUCCESS 状态失败: {e}')


def _mark_run_instance_error(run_uid: str) -> None:
    """Mark a run instance as failed when any task fails."""
    try:
        with Session(engine) as session:
            run = run_crud.get(session, run_uid)
            if run is None:
                return
            if run.status != Status.ERROR:
                workflow_runtime.update_run(
                    session,
                    run_uid,
                    RunUpdate(status=Status.ERROR, end_time=datetime.now(tz=UTC)),
                )
    except Exception as e:
        logger.warning(f'更新 RunInstance ERROR 状态失败: {e}')


def _flush_monitor_to_db(task_id: str) -> None:
    """将 Redis list task-monitor-{task_id} 原子 flush 到 DB, 然后删除 key."""
    redis = _get_redis()
    key = f'task-monitor-{task_id}'
    pipe = redis.pipeline()
    pipe.lrange(key, 0, -1)
    pipe.delete(key)
    logs, _ = pipe.execute()

    monitors = []
    for raw in logs:
        stat_data = json.loads(raw)
        if stat_data is not None:
            stat_data['task_uid'] = as_uuid(task_id)
            monitors.append(TaskMonitor(**stat_data))

    if monitors:
        with Session(engine) as session:
            workflow_runtime.insert_monitors(session, monitors)


# ---------------------------------------------------------------------------
# Public helpers used by task modules
# ---------------------------------------------------------------------------


def get_tool_data(tool_uid: str) -> ToolTaskInfo:
    """从数据库获取工具信息."""
    with Session(engine) as session:
        statement = select(ToolDefine).where(ToolDefine.uid == tool_uid).options(joinedload(ToolDefine.image))
        tool = session.scalar(statement)
        if not tool:
            error_msg = f'工具 {tool_uid} 不存在'
            raise RuntimeError(error_msg)
        if tool.image is None:
            error_msg = f'工具 {tool_uid} 未关联镜像'
            raise RuntimeError(error_msg)

        return ToolTaskInfo(
            uid=tool.uid,
            name=tool.name,
            file_mounts=tool.file_mounts,
            command_template=tool.command_template,
            dynamic_params=tool.dynamic_params,
            immutable_static_params=tool.immutable_static_params,
            modifiable_static_params=tool.modifiable_static_params,
            image=tool.image,
        )


def update_task(uid: str, data: TaskUpdate) -> bool:
    """更新任务状态."""
    try:
        with Session(engine) as session:
            workflow_runtime.update_task(session, uid, data)
            return True
    except Exception as e:
        logger.error(f'更新任务 {uid} 状态失败: {e}')
        return False


def get_tool_output(uid: str, k: str, max_retries: int = 5, retry_interval: float = 3.0) -> tuple[str, str | None]:
    """从数据库获取任务输出, 带轮询重试机制."""
    for attempt in range(max_retries):
        try:
            with Session(engine) as session:
                task = _require_task(session, uid)

                if task.status == Status.ERROR:
                    error_msg = f'任务 {uid} 执行失败, 无法获取输出'
                    raise ValueError(error_msg)  # noqa: TRY301

                if task.tool_output is not None:
                    if k == 'workdir' and task.base_dir:
                        result = str(Path(task.base_dir).absolute())
                        logger.debug(f'成功获取输出 {k}: {result}')
                        return result, task.tool_output.sample_name

                    all_files = task.tool_output.all_files()
                    if k in all_files:
                        result = str(all_files[k])
                        logger.debug(f'成功获取输出 {k}: {result}')
                        return result, task.tool_output.sample_name

                if attempt < max_retries - 1:
                    logger.debug(
                        f'任务 {uid} 的输出 {k} 尚未就绪 '
                        f'(状态: {task.status.name}, 尝试 {attempt + 1}/{max_retries}), 等待重试...'
                    )
                    time.sleep(retry_interval)
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f'获取任务输出失败 (尝试 {attempt + 1}/{max_retries}): {e}')
                time.sleep(retry_interval)
            else:
                raise

    err_msg = f'无法获取任务 {uid} 的输出 {k}, 已重试 {max_retries} 次'
    raise TimeoutError(err_msg)


def resolve_path(
    path_or_reference: str,
    session: Session,
    project_id: int | None,
    sample_uid: str | None,
) -> tuple[Path, str | None]:
    """解析路径引用为实际文件路径.

    Args:
        path_or_reference: 普通路径或引用 (如 "proj:index" 或 "sample:r1")
        session: 数据库会话
        project_id: 项目 ID, 用于项目级查找
        sample_uid: 样本UID, 用于获取样本文件

    Returns:
        解析后的文件路径

    Raises:
        ValueError: 如果引用无法解析 (在任务执行时抛出)

    Note:
        错误在运行时任务执行时抛出, 而非工作流验证时。
        这允许在某些映射尚未配置时创建工作流。
    """
    if not path_or_reference.startswith(('proj:', 'sample:')):
        return Path(path_or_reference), None

    if path_or_reference.startswith('proj:'):
        return _resolve_project_path(path_or_reference, session, project_id)

    if path_or_reference.startswith('sample:'):
        return _resolve_sample_path(path_or_reference, session, sample_uid)

    err_msg = f'Invalid path reference: {path_or_reference}'
    raise ValueError(err_msg)


def get_prefix(strings: list[str | None]) -> str | None:
    """获取两个字符串的共同前缀."""
    valid_strs = [s for s in strings if s is not None]
    if not valid_strs:
        return None

    pairs = zip(*valid_strs, strict=False)
    common_gen = takewhile(lambda x: all(c == x[0] for c in x), pairs)
    raw_prefix = ''.join(char_tuple[0] for char_tuple in common_gen)
    return raw_prefix.rstrip('.-_rR')


def _resolve_project_path(reference: str, session: Session, project_id: int | None) -> tuple[Path, str | None]:
    """解析 proj:<keyword> 为项目文件映射中的文件路径."""
    if project_id is None:
        err_msg = f'Cannot resolve {reference}: this is a temporary workflow. Use file path directly.'
        raise ValueError(err_msg)

    keyword = reference.split(':', 1)[1]
    mapping = project_file_mapping_crud.get_by_keyword(session, project_id, keyword)

    if not mapping:
        err_msg = f'Project file mapping not found: {keyword}'
        raise FileNotFoundError(err_msg)

    return Path(mapping.file_path).absolute(), keyword


def _resolve_sample_path(reference: str, session: Session, sample_uid: str | None) -> tuple[Path, str | None]:
    """解析 sample:<tag> 为样本文件路径."""
    if sample_uid is None:
        err_msg = f'Cannot resolve {reference}: this is a project-level workflow. Use proj:keyword references instead of sample:tag references.'
        raise ValueError(err_msg)

    tag = reference.split(':', 1)[1]
    sample = sample_crud.get_with_files(session, sample_uid)  # type: ignore

    if not sample:
        err_msg = f'Sample not found: {sample_uid}'
        raise KeyError(err_msg)

    matching_file = next((f for f in sample.sample_files if f.tag == tag), None)

    if not matching_file:
        err_msg = f'Sample file with tag "{tag}" not found'
        raise KeyError(err_msg)

    return Path(matching_file.file_path).absolute(), tag
