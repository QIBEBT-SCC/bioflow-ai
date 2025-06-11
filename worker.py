from celery import Celery

from app.core.config import get_settings

celery_app = Celery(
    'tasks', broker=get_settings().celery.BROKER, backend=get_settings().celery.BACKEND
)

celery_app.conf.update(
    worker_concurrency=4,
    task_serializer='pickle',
    result_serializer='pickle',
    accept_content=['pickle'],
    enable_utc=True,
    timezone='Asia/Shanghai',
)
celery_app.autodiscover_tasks(['app.tasks.run_tool', 'app.tasks.load_file'], None)


if __name__ == '__main__':
    args = ['worker', f'--loglevel={get_settings().logger.CELERY_LEVEL}']
    celery_app.worker_main(argv=args)
