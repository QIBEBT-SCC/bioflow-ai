import json
import platform
import socket
from datetime import datetime
from typing import Optional

import celery
import requests
from loguru import logger

from app.core.config import get_settings
from app.models.workflow import Status, ToolOutput
from app.schemas.task_instance import TaskUpdate
from app.schemas.tools import ToolTaskInfo


class BaseBioTask(celery.Task):
    def before_start(self, task_id, args, kwargs):
        os_info = platform.platform()
        hostname = socket.gethostname()
        data = TaskUpdate(status=Status.RUNNING, system=os_info, hostname=hostname, start_time=datetime.now())
        update_task(task_id, data)

    def on_success(self, retval, task_id, args, kwargs):
        data = TaskUpdate(tool_output=retval, status=Status.SUCCESS, end_time=datetime.now())
        update_task(task_id, data)
        update_monitor_log(task_id)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        data = TaskUpdate(status=Status.ERROR, end_time=datetime.now())
        update_task(task_id, data)
        update_monitor_log(task_id)


def get_tool_data(tool_uid: str) -> ToolTaskInfo:
    url = f'http://localhost:{get_settings().network.PORT}/api/v1/tools/{tool_uid}/task_info'
    headers = {'Content-Type': 'application/json'}

    response = requests.get(url, headers=headers)
    assert response.status_code == 200

    return ToolTaskInfo.model_validate(response.json())


def update_task(uid: str, data: TaskUpdate, max_retries: int = 3) -> bool:
    """更新任务状态，带重试机制
    
    Args:
        uid: 任务 UID
        data: 更新数据
        max_retries: 最大重试次数
        
    Returns:
        bool: 是否更新成功
    """
    url = f'http://localhost:{get_settings().network.PORT}/api/v1/tasks/{uid}'
    headers = {'Content-Type': 'application/json'}
    json_str = data.model_dump_json(exclude_unset=True)
    
    for attempt in range(max_retries):
        try:
            response = requests.patch(
                url, 
                json=json.loads(json_str), 
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return True
            else:
                logger.warning(
                    f'更新任务状态失败 (尝试 {attempt + 1}/{max_retries}): '
                    f'HTTP {response.status_code} - {response.text}'
                )
                
        except requests.exceptions.Timeout:
            logger.warning(f'更新任务状态超时 (尝试 {attempt + 1}/{max_retries})')
        except requests.exceptions.ConnectionError as e:
            logger.warning(f'连接失败 (尝试 {attempt + 1}/{max_retries}): {e}')
        except Exception as e:
            logger.error(f'更新任务状态异常 (尝试 {attempt + 1}/{max_retries}): {e}')
            
        # 如果不是最后一次尝试，等待一段时间后重试
        if attempt < max_retries - 1:
            import time
            time.sleep(1 * (attempt + 1))  # 递增等待时间
    
    logger.error(f'更新任务 {uid} 状态失败，已达到最大重试次数')
    return False


def update_monitor_log(uid: str, max_retries: int = 3) -> bool:
    """更新监控日志，带重试机制
    
    Args:
        uid: 任务 UID
        max_retries: 最大重试次数
        
    Returns:
        bool: 是否更新成功
    """
    url = f'http://localhost:{get_settings().network.PORT}/api/v1/tasks/{uid}/notify'
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, timeout=10)
            
            if response.status_code == 200:
                return True
            else:
                logger.warning(
                    f'更新监控日志失败 (尝试 {attempt + 1}/{max_retries}): '
                    f'HTTP {response.status_code}'
                )
                
        except requests.exceptions.Timeout:
            logger.warning(f'更新监控日志超时 (尝试 {attempt + 1}/{max_retries})')
        except Exception as e:
            logger.error(f'更新监控日志异常 (尝试 {attempt + 1}/{max_retries}): {e}')
            
        if attempt < max_retries - 1:
            import time
            time.sleep(1)
    
    logger.error(f'更新任务 {uid} 监控日志失败，已达到最大重试次数')
    return False


def get_tool_output(uid: str, k: str) -> str:
    url = f'http://localhost:{get_settings().network.PORT}/api/v1/tasks/{uid}/result'
    headers = {'Content-Type': 'application/json'}

    response = requests.get(url, headers=headers)
    assert response.status_code == 200

    task_output = ToolOutput.model_validate(response.json())
    return str(task_output.all_files()[k])
