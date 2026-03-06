import io
from contextlib import asynccontextmanager
from functools import lru_cache

import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.responses import Response
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
from app.core.s3_client import init_bucket
from app.core.session import create_db_and_tables
from app.utils.logger import init_logging

init_logging()


@asynccontextmanager
async def lifespan(main_app: FastAPI):
    create_db_and_tables()
    init_bucket()
    yield

    logger.info('server stop')


app = FastAPI(title=get_settings().PROJECT_NAME, version=get_settings().VERSION, lifespan=lifespan)

# 允许所有源地址
origins = [
    'http://localhost:3000',
    'http://172.18.19.113:3000',
]


class LogRequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.debug(f'Received request: {await request.json()}')
        response = await call_next(request)
        return response


app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=['*'], allow_headers=['*']
)
# app.add_middleware(LogRequestMiddleware)


@app.get('/openapi.yaml', include_in_schema=False)
@lru_cache()
def read_openapi_yaml() -> Response:
    openapi_json = app.openapi()
    yaml_s = io.StringIO()
    yaml.dump(openapi_json, yaml_s)
    return Response(yaml_s.getvalue(), media_type='text/yaml')


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


def main() -> None:
    config = uvicorn.Config(
        'api:app', host=get_settings().network.HOST, port=get_settings().network.PORT, access_log=True, workers=1
    )
    server = uvicorn.Server(config)

    try:
        server.run()
    except KeyboardInterrupt:
        logger.bind().error('server closed by user')


if __name__ == '__main__':
    main()
