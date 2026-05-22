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
from sqlalchemy.orm import joinedload
from sqlmodel import Session, select

import app.crud.run_instance as run_crud
import app.crud.task_instance as task_crud
import app.crud.task_monitor as monitor_crud
from app.core.session import engine
from app.crud import project as project_crud
from app.crud.resource import sample as sample_crud
from app.models.tools import ToolDefine
from app.models.workflow import Status, TaskMonitor
from app.schemas.run_instance import RunUpdate
from app.schemas.task_instance import TaskUpdate
from app.schemas.tools import ToolTaskInfo


class BaseBioTask(celery.Task):
    def before_start(self, task_id: str, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """任务开始前的钩子 - 更新 DB 状态并维护 Redis 计数."""
        try:
            with Session(engine) as session:
                task = task_crud.get(session, task_id)
                task.status = Status.RUNNING
                task.system = platform.platform()
                task.hostname = socket.gethostname()
                task.start_time = datetime.now(UTC)
                session.add(task)
                session.commit()
                logger.info(f'Task {task.name}({task_id}) START')

                instance_uid = str(task.instance_uid)

            _update_run_stats(instance_uid, Status.WAITING, Status.RUNNING)
            _check_and_update_run_instance_running(instance_uid)
        except Exception as e:
            logger.exception(f'更新任务 {task_id} 启动状态失败: {e}')

    def on_success(self, retval, task_id, args, kwargs) -> None:  # noqa: ANN001
        """任务成功后的钩子 - 更新 DB 状态、Redis 计数、flush 监控日志."""
        try:
            with Session(engine) as session:
                task = task_crud.get(session, task_id)
                task.status = Status.SUCCESS
                task.tool_output = retval
                task.end_time = datetime.now(UTC)
                session.add(task)
                session.commit()
                logger.success(f'Task {task.name}({task_id}) SUCCESS')

                instance_uid = str(task.instance_uid)

            _update_run_stats(instance_uid, Status.RUNNING, Status.SUCCESS)
            _check_and_update_run_instance_success(instance_uid)
        except Exception as e:
            logger.exception(f'更新任务 {task_id} 成功状态失败: {e}')

        try:
            _flush_monitor_to_db(task_id)
        except Exception as e:
            logger.warning(f'flush 监控日志失败: {e}')

    def on_failure(self, exc, task_id, args, kwargs, einfo) -> None:  # noqa: ANN001
        """任务失败后的钩子 - 更新 DB 状态、Redis 计数、flush 监控日志."""
        try:
            with Session(engine) as session:
                task = task_crud.get(session, task_id)
                task.status = Status.ERROR
                task.end_time = datetime.now(UTC)
                session.add(task)
                session.commit()
                logger.error(f'Task {task.name}({task_id}) ERROR')

                instance_uid = str(task.instance_uid)

            _update_run_stats(instance_uid, Status.RUNNING, Status.ERROR)
            _check_and_update_run_instance_error(instance_uid)
        except Exception as e:
            logger.error(f'更新任务 {task_id} 失败状态失败: {e}')

        try:
            _flush_monitor_to_db(task_id)
        except Exception as e:
            logger.warning(f'flush 监控日志失败: {e}')


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------


def _get_redis() -> Redis:
    from script.worker import monitor_redis_client

    return monitor_redis_client


def _ensure_run_stats(redis_client: Redis, run_uid: str) -> str:
    """确保 Redis Hash run-stats-{run_uid} 存在, 不存在则从 DB 初始化."""
    key = f'run-stats-{run_uid}'
    if not redis_client.exists(key):
        with Session(engine) as session:
            stats = run_crud.get_run_task_stats(session, run_uid)
        redis_client.hset(
            key,
            mapping={
                'total': stats.total,
                'waiting': stats.waiting or 0,
                'running': stats.running or 0,
                'success': stats.success or 0,
                'error': stats.error or 0,
            },
        )
        redis_client.expire(key, 86400)
    return key


def _update_run_stats(run_uid: str, from_status: Status, to_status: Status) -> None:
    """原子更新 Redis Hash 中的运行统计计数."""
    try:
        redis = _get_redis()
        key = _ensure_run_stats(redis, run_uid)
        redis.hincrby(key, from_status.name.lower(), -1)
        redis.hincrby(key, to_status.name.lower(), 1)
    except Exception as e:
        logger.warning(f'更新 Redis run-stats 失败: {e}')


def _check_and_update_run_instance_running(run_uid: str) -> None:
    """如果是第一个 RUNNING 任务, 将 RunInstance 状态更新为 RUNNING."""
    try:
        redis = _get_redis()
        key = f'run-stats-{run_uid}'
        running_count = redis.hget(key, 'running')
        if running_count and int(running_count) == 1:  # type: ignore
            with Session(engine) as session:
                run_crud.update(session, run_uid, RunUpdate(status=Status.RUNNING, start_time=datetime.now(tz=UTC)))
    except Exception as e:
        logger.warning(f'更新 RunInstance RUNNING 状态失败: {e}')


def _check_and_update_run_instance_success(run_uid: str) -> None:
    """如果所有任务都 SUCCESS, 将 RunInstance 标记为 SUCCESS."""
    try:
        redis = _get_redis()
        key = f'run-stats-{run_uid}'
        stats = redis.hgetall(key)
        total = int(stats.get(b'total', 0))
        success = int(stats.get(b'success', 0))
        if 0 < total == success:
            with Session(engine) as session:
                run_crud.update(session, run_uid, RunUpdate(status=Status.SUCCESS, end_time=datetime.now(tz=UTC)))
    except Exception as e:
        logger.warning(f'更新 RunInstance SUCCESS 状态失败: {e}')


def _check_and_update_run_instance_error(run_uid: str) -> None:
    """有任务失败时, 将 RunInstance 标记为 ERROR."""
    try:
        with Session(engine) as session:
            run_crud.update(session, run_uid, RunUpdate(status=Status.ERROR, end_time=datetime.now(tz=UTC)))
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
            stat_data['task_uid'] = task_id
            monitors.append(TaskMonitor.model_validate(stat_data))

    if monitors:
        with Session(engine) as session:
            monitor_crud.insert_list(session, monitors)


# ---------------------------------------------------------------------------
# Public helpers used by task modules
# ---------------------------------------------------------------------------


def get_tool_data(tool_uid: str) -> ToolTaskInfo:
    """从数据库获取工具信息."""
    with Session(engine) as session:
        statement = select(ToolDefine).where(ToolDefine.uid == tool_uid).options(joinedload(ToolDefine.image))  # type: ignore
        tool = session.exec(statement).one_or_none()
        if not tool:
            error_msg = f'工具 {tool_uid} 不存在'
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
            task_crud.update(session, uid, data)
            return True
    except Exception as e:
        logger.error(f'更新任务 {uid} 状态失败: {e}')
        return False


def get_tool_output(uid: str, k: str, max_retries: int = 5, retry_interval: float = 3.0) -> tuple[str, str | None]:
    """从数据库获取任务输出, 带轮询重试机制."""
    for attempt in range(max_retries):
        try:
            with Session(engine) as session:
                task = task_crud.get(session, uid)

                if task.status == Status.ERROR:
                    error_msg = f'任务 {uid} 执行失败, 无法获取输出'
                    raise ValueError(error_msg)  # noqa: TRY301

                if task.tool_output is not None:
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
    mapping = project_crud.get_file_mapping(session, project_id, keyword)

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
