import celery
from loguru import logger


class BaseBioTask(celery.Task):
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f'{retval} success: {task_id}')
