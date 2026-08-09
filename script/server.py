from contextlib import asynccontextmanager
from importlib.metadata import metadata

import uvicorn
from fastapi import FastAPI
from loguru import logger
from starlette.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import (
    agent,
    auth,
    code,
    genome,
    notification,
    project,
    resource,
    run_instance,
    setting,
    skill,
    task_instance,
    tool,
    user,
    workflow,
)
from app.core.config import get_settings
from app.core.redis_client import close_cache_regis, notification_redis_client, setup_cache_redis
from app.core.s3_client import init_bucket
from app.core.session import create_db_and_tables
from app.services.notification.gateway import NotificationGateway
from app.utils.middleware import LogRequestMiddleware

meta = metadata('bioflow-ai')


@asynccontextmanager
async def lifespan(main_app: FastAPI):  # noqa: ANN201, D103
    create_db_and_tables()
    init_bucket()
    main_app.state.redis = await setup_cache_redis()
    notification_gateway = NotificationGateway(notification_redis_client)
    await notification_gateway.start()
    main_app.state.notification_gateway = notification_gateway

    yield

    await notification_gateway.close()
    await notification_redis_client.aclose()
    await close_cache_regis(main_app.state.redis)
    logger.info('server stop')


def create_app(*, log_requests: bool = False) -> FastAPI:
    """Create the FastAPI app."""
    app = FastAPI(
        title=meta['Name'],
        version=meta['Version'],
        description=meta['Summary'],
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().network.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )
    if log_requests:
        app.add_middleware(LogRequestMiddleware)

    app.include_router(auth.router, prefix='/api/v1/auth', tags=['auth'])
    app.include_router(agent.session_router, prefix='/api/v1/agent-sessions', tags=['agent'])
    app.include_router(agent.run_router, prefix='/api/v1/agent-runs', tags=['agent'])
    app.include_router(user.router, prefix='/api/v1/user', tags=['user'])
    app.include_router(setting.router, prefix='/api/v1/settings', tags=['setting'])
    app.include_router(notification.router, prefix='/api/v1/notifications', tags=['notification'])
    app.include_router(skill.router, prefix='/api/v1/skills', tags=['skill'])
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
    app.include_router(code.router, prefix='/api/v1/codes', tags=['code'])
    app.include_router(resource.db_router, prefix='/api/v1/bio_dbs', tags=['resource'])
    app.include_router(genome.reference_router, prefix='/api/v1/references', tags=['genome'])
    app.include_router(resource.sample_router, prefix='/api/v1/samples', tags=['resource'])
    return app


app = create_app()


def main(*, log_requests: bool = False) -> None:  # noqa: D103
    config = uvicorn.Config(
        create_app(log_requests=log_requests),
        host=get_settings().network.HOST,
        port=get_settings().network.PORT,
        access_log=True,
        workers=1,
        log_config=None,
    )
    server = uvicorn.Server(config)

    try:
        server.run()
    except KeyboardInterrupt:
        logger.bind().error('server closed by user')


if __name__ == '__main__':
    main()
