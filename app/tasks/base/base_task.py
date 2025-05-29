import json
from datetime import datetime

import celery
import requests

from app.core.config import get_settings
from app.models.tools import ToolTaskInfo
from app.models.workflow import TaskUpdate, Status


class BaseBioTask(celery.Task):
    def before_start(self, task_id, args, kwargs):
        data = TaskUpdate(status=Status.RUNNING, start_time=datetime.now())
        update_task(task_id, data)

    def on_success(self, retval, task_id, args, kwargs):
        data = TaskUpdate(result=retval, status=Status.SUCCESS, end_time=datetime.now())
        update_task(task_id, data)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        data = TaskUpdate(status=Status.ERROR, end_time=datetime.now())
        update_task(task_id, data)


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


def get_tool_output(uid: str, k: str) -> str:
    url = f'http://localhost:{get_settings().network.PORT}/api/v1/tasks/{uid}/result'
    headers = {'Content-Type': 'application/json'}

    response = requests.get(url, headers=headers)
    assert response.status_code == 200

    return response.json()[k]
