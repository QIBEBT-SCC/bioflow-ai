import json
import platform
import socket
from datetime import datetime

import celery
import requests

from app.core.config import get_settings
from app.schemas.tools import ToolTaskInfo
from app.models.workflow import Status, ToolOutput
from app.schemas.task_instance import TaskUpdate


class BaseBioTask(celery.Task):
    def before_start(self, task_id, args, kwargs):
        os_info = platform.platform()
        hostname = socket.gethostname()
        data = TaskUpdate(
            status=Status.RUNNING, system=os_info, hostname=hostname, start_time=datetime.now()
        )
        update_task(task_id, data)

    def on_success(self, retval, task_id, args, kwargs):
        data = TaskUpdate(tool_output=retval, status=Status.SUCCESS, end_time=datetime.now())
        update_task(task_id, data)
        update_monitor_log(task_id)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        data = TaskUpdate(status=Status.ERROR, end_time=datetime.now())
        update_task(task_id, data)
        update_monitor_log(task_id)


def get_tool_data(uid: str) -> ToolTaskInfo:
    url = f'http://localhost:{get_settings().network.PORT}/api/v1/tools/{uid}/task_info'
    headers = {'Content-Type': 'application/json'}

    response = requests.get(url, headers=headers)
    assert response.status_code == 200

    return ToolTaskInfo.model_validate(response.json())


def update_task(uid: str, data: TaskUpdate):
    url = f'http://localhost:{get_settings().network.PORT}/api/v1/tasks/{uid}'
    headers = {'Content-Type': 'application/json'}
    json_str = data.model_dump_json(exclude_unset=True)

    response = requests.patch(url, json=json.loads(json_str), headers=headers)
    assert response.status_code == 200


def update_monitor_log(uid: str):
    url = f'http://localhost:{get_settings().network.PORT}/api/v1/tasks/{uid}/notify'
    response = requests.post(url)
    assert response.status_code == 200


def get_tool_output(uid: str, k: str) -> str:
    url = f'http://localhost:{get_settings().network.PORT}/api/v1/tasks/{uid}/result'
    headers = {'Content-Type': 'application/json'}

    response = requests.get(url, headers=headers)
    assert response.status_code == 200

    task_output = ToolOutput.model_validate(response.json())
    return str(task_output.all_files()[k])
