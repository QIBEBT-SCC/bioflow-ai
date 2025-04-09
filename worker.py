from celery import Celery

celery_app = Celery(
    'tasks',
    backend='rpc://single_cell:single_cell@localhost:5672//',
    broker='pyamqp://single_cell:single_cell@localhost:5672//',
)

celery_app.conf.update(enable_utc=True, timezone='Asia/Shanghai')

if __name__ == '__main__':
    args = ['worker', '--concurrency=1', '--loglevel=INFO']
    celery_app.worker_main(argv=args)
