from celery import Celery

celery_app = Celery(
    'tasks', broker='amqp://bioflow_ai:123456@localhost:5672//', backend='redis://localhost:6379/0'
)

celery_app.conf.update(
    worker_concurrency=2,
    task_serializer='pickle',
    result_serializer='pickle',
    accept_content=['pickle'],
    enable_utc=True,
    timezone='Asia/Shanghai',
)
celery_app.autodiscover_tasks(['app.tasks.run_tool', 'app.tasks.load_file'], None)


if __name__ == '__main__':
    args = ['worker', '--loglevel=INFO']
    celery_app.worker_main(argv=args)
