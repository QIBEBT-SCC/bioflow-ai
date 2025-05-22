import celery
from loguru import logger


class BaseBioTask(celery.Task):
    def before_start(self, task_id, args, kwargs):
        logger.info(task_id)

    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f'success: {task_id}')


def get_tool_output(uid: str, k: str) -> str:
    raise NotImplementedError
