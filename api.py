from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from loguru import logger
from starlette.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import workflow, tool, auth, project, user, run_instance, task_instance
from app.core.session import create_db_and_tables
from app.core.config import get_settings
from app.utils.logger import init_logging


@asynccontextmanager
async def lifespan(main_app: FastAPI):
    create_db_and_tables()
    yield

    logger.info('server stop')


app = FastAPI(title=get_settings().PROJECT_NAME, version=get_settings().VERSION, lifespan=lifespan)

origins = ['http://localhost', 'http://127.0.0.1']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(auth.router, prefix='/api/v1/auth', tags=['auth'])
app.include_router(user.router, prefix='/api/v1/user', tags=['user'])
app.include_router(project.tag_router, prefix='/api/v1/project-tags', tags=['project'])
app.include_router(project.router, prefix='/api/v1/projects', tags=['project'])
app.include_router(workflow.router, prefix='/api/v1/workflows', tags=['workflow'])
app.include_router(run_instance.router, prefix='/api/v1/runs', tags=['runs'])
app.include_router(task_instance.router, prefix='/api/v1/tasks', tags=['tasks'])
app.include_router(tool.router, prefix='/api/v1/tools', tags=['tool'])
app.include_router(tool.tag_router, prefix='/api/v1/tool-tags', tags=['tool'])
app.include_router(tool.group_router, prefix='/api/v1/tool-groups', tags=['tool'])


def main() -> None:
    config = uvicorn.Config(
        'api:app',
        host=get_settings().network.HOST,
        port=get_settings().network.PORT,
        access_log=True,
        workers=1,
    )
    server = uvicorn.Server(config)
    init_logging()

    try:
        server.run()
    except KeyboardInterrupt:
        logger.bind().error('server closed by user')


if __name__ == '__main__':
    main()
