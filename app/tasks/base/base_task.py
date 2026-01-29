import platform
import socket
from datetime import datetime

import celery
from loguru import logger
from sqlalchemy.orm import joinedload
from sqlmodel import Session, select

import app.crud.task_instance as task_crud
from app.core.session import engine
from app.models.tools import ToolDefine
from app.models.workflow import Status
from app.schemas.task_instance import TaskUpdate
from app.schemas.tools import ToolTaskInfo


class BaseBioTask(celery.Task):
    def before_start(self, task_id, args, kwargs):
        """任务开始前的钩子 - 直接更新数据库"""
        try:
            with Session(engine) as session:
                task = task_crud.get(session, task_id)
                task.status = Status.RUNNING
                task.system = platform.platform()
                task.hostname = socket.gethostname()
                task.start_time = datetime.now()

                session.add(task)
                session.commit()
                logger.debug(f'任务 {task_id} 状态已更新为 RUNNING')
        except Exception as e:
            logger.error(f'更新任务 {task_id} 启动状态失败: {e}')

    def on_success(self, retval, task_id, args, kwargs):
        """任务成功后的钩子 - 直接更新数据库"""
        try:
            with Session(engine) as session:
                task = task_crud.get(session, task_id)
                task.status = Status.SUCCESS
                task.tool_output = retval
                task.end_time = datetime.now()

                session.add(task)
                session.commit()
                logger.success(f'任务 {task_id} 状态已更新为 SUCCESS')
        except Exception as e:
            logger.error(f'更新任务 {task_id} 成功状态失败: {e}')

        # 更新 Redis 监控日志（用于实时推送）
        try:
            update_monitor_redis(task_id)
        except Exception as e:
            logger.warning(f'更新 Redis 监控日志失败: {e}')

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """任务失败后的钩子 - 直接更新数据库"""
        try:
            with Session(engine) as session:
                task = task_crud.get(session, task_id)
                task.status = Status.ERROR
                task.end_time = datetime.now()

                session.add(task)
                session.commit()
                logger.debug(f'任务 {task_id} 状态已更新为 ERROR')
        except Exception as e:
            logger.error(f'更新任务 {task_id} 失败状态失败: {e}')

        # 更新 Redis 监控日志（用于实时推送）
        try:
            update_monitor_redis(task_id)
        except Exception as e:
            logger.warning(f'更新 Redis 监控日志失败: {e}')


def get_tool_data(tool_uid: str) -> ToolTaskInfo:
    """从数据库获取工具信息

    Args:
        tool_uid: 工具 UID

    Returns:
        ToolTaskInfo: 工具任务信息
    """
    with Session(engine) as session:
        statement = select(ToolDefine).where(ToolDefine.uid == tool_uid).options(joinedload(ToolDefine.image))

        tool = session.exec(statement).one_or_none()
        if not tool:
            raise ValueError(f'工具 {tool_uid} 不存在')

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
    """更新任务状态

    Args:
        uid: 任务 UID
        data: 更新数据

    Returns:
        bool: 是否更新成功
    """
    try:
        with Session(engine) as session:
            task_crud.update(session, uid, data)
            return True
    except Exception as e:
        logger.error(f'更新任务 {uid} 状态失败: {e}')
        return False


def update_monitor_redis(uid: str) -> bool:
    """更新 Redis 监控日志（用于实时通知）

    Args:
        uid: 任务 UID

    Returns:
        bool: 是否更新成功
    """
    try:
        from worker import monitor_redis_client

        # 从数据库获取任务信息
        with Session(engine) as session:
            task = task_crud.get(session, uid)

            # 更新 Redis 中的运行实例统计
            run_stats_key = f'run-stats-{task.instance_uid}'

            if task.status == Status.RUNNING:
                monitor_redis_client.hincrby(run_stats_key, 'waiting', -1)
                monitor_redis_client.hincrby(run_stats_key, 'running', 1)
            elif task.status == Status.SUCCESS:
                monitor_redis_client.hincrby(run_stats_key, 'running', -1)
                monitor_redis_client.hincrby(run_stats_key, 'success', 1)
            elif task.status == Status.ERROR:
                monitor_redis_client.hincrby(run_stats_key, 'running', -1)
                monitor_redis_client.hincrby(run_stats_key, 'error', 1)

            return True
    except Exception as e:
        logger.error(f'更新 Redis 监控日志失败: {e}')
        return False


def get_tool_output(uid: str, k: str, max_retries: int = 30, retry_interval: float = 2.0) -> str:
    """从数据库获取任务输出，带轮询重试机制

    用于处理任务间的依赖关系。如果输出尚未就绪，会进行轮询重试。

    Args:
        uid: 任务 UID
        k: 输出字段名称
        max_retries: 最大重试次数（默认 30 次）
        retry_interval: 重试间隔（秒，默认 2 秒）

    Returns:
        str: 任务输出的文件路径

    Raises:
        TimeoutError: 超过最大重试次数仍未获取到输出
        ValueError: 任务不存在或输出字段不存在
    """
    import time

    for attempt in range(max_retries):
        try:
            with Session(engine) as session:
                task = task_crud.get(session, uid)

                # 检查任务状态
                if task.status == Status.ERROR:
                    raise ValueError(f'任务 {uid} 执行失败，无法获取输出')

                # 检查输出是否存在
                if task.tool_output is not None:
                    all_files = task.tool_output.all_files()
                    if k in all_files:
                        result = str(all_files[k])
                        logger.debug(f'成功获取输出 {k}: {result}')
                        return result

                # 输出尚未就绪，等待重试
                if attempt < max_retries - 1:
                    logger.debug(
                        f'任务 {uid} 的输出 {k} 尚未就绪 '
                        f'(状态: {task.status.name}, 尝试 {attempt + 1}/{max_retries})，等待重试...'
                    )
                    time.sleep(retry_interval)

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f'获取任务输出失败 (尝试 {attempt + 1}/{max_retries}): {e}')
                time.sleep(retry_interval)
            else:
                raise

    raise TimeoutError(
        f'无法获取任务 {uid} 的输出 {k}，已重试 {max_retries} 次（总等待时间: {max_retries * retry_interval:.0f}秒）'
    )
