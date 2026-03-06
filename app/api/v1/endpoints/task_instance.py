import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

import app.crud.run_instance as run_crud
import app.crud.task_instance as task_crud
import app.crud.task_monitor as monitor_crud
from app.core.redis_client import monitor_redis_client
from app.core.session import SessionDep
from app.models.workflow import Status, TaskInstance, TaskMonitor, ToolOutput
from app.schemas.run_instance import RunUpdate
from app.schemas.task_instance import MonitorPublic, SimpleTaskPublic, TaskPublic, TaskUpdate
from app.utils.security import UserDep

router = APIRouter()


@router.get('')
def get_all_tasks(
    session: SessionDep,
    current_user: UserDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> list[SimpleTaskPublic]:
    """Get a paginated list of task instances for the current user."""
    return task_crud.get_all(session, limit, offset, current_user.id)


@router.get('/count')
def get_task_count(session: SessionDep, current_user: UserDep) -> int:
    """Get the total count of task instances for the current user."""
    return task_crud.get_count(session, current_user.id)


@router.get('/recent/{hour}')
def get_by_time(hour: int, session: SessionDep, current_user: UserDep) -> list[SimpleTaskPublic]:
    """Get task instances created within the last N hours."""
    return task_crud.get_all_by_time(session, current_user.id, timedelta(hours=hour))


@router.patch('/{uid}')
async def update_task(uid: str, task_update: TaskUpdate, session: SessionDep) -> TaskInstance:
    """更新任务实例状态.

    使用 Redis Hash 原子操作更新统计信息, 避免竞态条件.
    """
    updated_task = task_crud.update(session, uid, task_update)

    run_uid = str(updated_task.instance_uid)
    redis_key = f'run-stats-{run_uid}'

    # 检查 Redis 中是否存在统计信息
    if not await monitor_redis_client.exists(redis_key):
        # 如果不存在, 从数据库重新计算并初始化
        run_cache_data = run_crud.get_run_task_stats(session, run_uid)
        await monitor_redis_client.hset(
            redis_key,
            mapping={
                'total': run_cache_data.total,
                'waiting': run_cache_data.waiting or 0,
                'running': run_cache_data.running or 0,
                'success': run_cache_data.success or 0,
                'error': run_cache_data.error or 0,
            },
        )
        await monitor_redis_client.expire(redis_key, 86400)

    # 使用原子操作更新统计信息
    if task_update.status:
        match task_update.status:
            case Status.RUNNING:
                # waiting -> running
                await monitor_redis_client.hincrby(redis_key, 'waiting', -1)
                await monitor_redis_client.hincrby(redis_key, 'running', 1)

                # 获取当前 running 数量判断是否需要更新运行实例状态
                running_count = await monitor_redis_client.hget(redis_key, 'running')
                if running_count and int(running_count) == 1:
                    run_crud.update(
                        session,
                        run_uid,
                        RunUpdate(
                            status=Status.RUNNING,
                            start_time=task_update.start_time if task_update.start_time else datetime.now(tz=UTC),
                        ),
                    )

            case Status.SUCCESS:
                # running -> success
                await monitor_redis_client.hincrby(redis_key, 'running', -1)
                await monitor_redis_client.hincrby(redis_key, 'success', 1)

                # 获取统计信息判断是否所有任务都成功
                stats = await monitor_redis_client.hgetall(redis_key)
                total = int(stats.get(b'total', 0))
                success = int(stats.get(b'success', 0))

                if success == total:
                    run_crud.update(
                        session,
                        run_uid,
                        RunUpdate(
                            status=Status.SUCCESS,
                            end_time=task_update.end_time if task_update.end_time else datetime.now(tz=UTC),
                        ),
                    )

            case Status.ERROR:
                # running -> error
                await monitor_redis_client.hincrby(redis_key, 'running', -1)
                await monitor_redis_client.hincrby(redis_key, 'error', 1)

                # 有错误就将运行实例标记为错误
                run_crud.update(
                    session,
                    run_uid,
                    RunUpdate(
                        status=Status.ERROR,
                        end_time=task_update.end_time if task_update.end_time else datetime.now(tz=UTC),
                    ),
                )

    return updated_task


@router.get('/{uid}')
def get_task_instance(uid: str, session: SessionDep) -> TaskPublic:
    """Get a task instance by its UID."""
    return task_crud.get(session, uid)


@router.get('/{uid}/result')
def get_task_result(uid: str, session: SessionDep) -> ToolOutput:
    """Get the tool output result of a task instance."""
    task = task_crud.get(session, uid)
    return task.tool_output


@router.post('/{uid}/notify')
async def read_resource_monitor(uid: str, session: SessionDep) -> None:
    """Flush Redis monitor logs for a task into the database."""
    key = f'task-monitor-{uid}'
    pipe = monitor_redis_client.pipeline()
    pipe.lrange(key, 0, -1)
    pipe.delete(key)
    logs, _ = await pipe.execute()

    stats_list = [json.loads(log) for log in logs]

    monitors = []
    for stat_data in stats_list:
        if stat_data is not None:
            stat_data['task_uid'] = uid
            monitors.append(TaskMonitor.model_validate(stat_data))

    monitor_crud.insert_list(session, monitors)


@router.get('/{uid}/monitor')
def get_monitor_log(uid: str, session: SessionDep) -> list[MonitorPublic]:
    """Get monitor log entries for a task instance."""
    return monitor_crud.get(session, uid)


@router.get('/{uid}/log')
def get_task_log(uid: str, session: SessionDep) -> dict:
    """Get the text log file content for a task instance."""
    task = task_crud.get(session, uid)

    log_path = task.tool_output.log  # This will be replaced with actual log path
    if log_path and Path(log_path).exists():
        with Path(log_path).open('r', encoding='utf-8') as f:
            lines = f.read()

            return {'content': lines}
    else:
        raise HTTPException(status_code=404, detail='Log file not found')
