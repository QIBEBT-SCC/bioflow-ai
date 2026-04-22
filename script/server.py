from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request

from app.api.v1.endpoints import (
    auth,
    chat,
    genome,
    project,
    resource,
    run_instance,
    setting,
    task_instance,
    tool,
    user,
    workflow,
)
from app.core.config import get_settings
from app.core.redis_client import close_cache_regis, setup_cache_redis
from app.core.s3_client import init_bucket
from app.core.session import create_db_and_tables
from app.utils.logger import init_logging
from app.utils.middleware.csrf import CSRFMiddleware

init_logging()


@asynccontextmanager
async def lifespan(main_app: FastAPI):  # noqa: ANN201, D103
    create_db_and_tables()
    init_bucket()
    main_app.state.redis = await setup_cache_redis()

    yield

    await close_cache_regis(main_app.state.redis)
    logger.info('server stop')


app = FastAPI(title=get_settings().PROJECT_NAME, version=get_settings().VERSION, lifespan=lifespan)

# 允许所有源地址
origins = [
    'http://localhost:3000',
    'http://172.18.19.113:3000',
]


class LogRequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # noqa: ANN001, ANN201, D102
        logger.debug(f'Received request: {await request.json()}')
        return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*', 'X-CSRF-Token'],
)
app.add_middleware(CSRFMiddleware)
# app.add_middleware(LogRequestMiddleware)


app.include_router(auth.router, prefix='/api/v1/auth', tags=['auth'])
app.include_router(user.router, prefix='/api/v1/user', tags=['user'])
app.include_router(setting.router, prefix='/api/v1/settings', tags=['setting'])
app.include_router(project.tag_router, prefix='/api/v1/project-tags', tags=['project'])
app.include_router(project.router, prefix='/api/v1/projects', tags=['project'])
app.include_router(workflow.router, prefix='/api/v1/workflows', tags=['workflow'])
app.include_router(run_instance.router, prefix='/api/v1/runs', tags=['runs'])
app.include_router(task_instance.router, prefix='/api/v1/tasks', tags=['tasks'])
app.include_router(tool.router, prefix='/api/v1/tools', tags=['tool'])
app.include_router(tool.tag_router, prefix='/api/v1/tool-tags', tags=['tool'])
app.include_router(tool.group_router, prefix='/api/v1/tool-groups', tags=['tool'])
app.include_router(tool.image_router, prefix='/api/v1/images', tags=['tool'])
app.include_router(tool.doc_router, prefix='/api/v1/documents', tags=['tool'])
app.include_router(resource.db_router, prefix='/api/v1/bio_dbs', tags=['resource'])
app.include_router(genome.reference_router, prefix='/api/v1/references', tags=['genome'])
app.include_router(resource.sample_router, prefix='/api/v1/samples', tags=['resource'])
app.include_router(chat.router, prefix='/api/v1/chat', tags=['chat'])


def main() -> None:  # noqa: D103
    config = uvicorn.Config(
        'script.server:app',
        host=get_settings().network.HOST,
        port=get_settings().network.PORT,
        access_log=True,
        workers=1,
    )
    server = uvicorn.Server(config)

    try:
        server.run()
    except KeyboardInterrupt:
        logger.bind().error('server closed by user')


if __name__ == '__main__':
    main()
