from contextlib import asynccontextmanager
from importlib.metadata import metadata

import uvicorn
from fastapi import Depends, FastAPI
from loguru import logger
from starlette.middleware.cors import CORSMiddleware

from bioflow_ai.api.middleware import AuthCookieCleanupMiddleware, LogRequestMiddleware
from bioflow_ai.api.v1.endpoints import (
    agent,
    auth,
    code,
    code_agent,
    genome,
    notification,
    project,
    resource,
    setting,
    skill,
    tool,
    tool_memory,
    user,
    workflow,
)
from bioflow_ai.api.v2.endpoints import workflow as workflow_v2
from bioflow_ai.core.auth import get_current_active_user, require_role
from bioflow_ai.core.authorization import authorize_project_request, authorize_workflow_request
from bioflow_ai.core.config import get_settings
from bioflow_ai.infrastructure.database import create_db_and_tables
from bioflow_ai.infrastructure.object_storage.client import init_bucket
from bioflow_ai.infrastructure.redis.client import (
    close_cache_regis,
    notification_redis_client,
    setup_cache_redis,
    wait_for_cache_redis,
)
from bioflow_ai.llm.checkpoint import open_checkpointer
from bioflow_ai.schemas.user_config import UserRole
from bioflow_ai.services.notification.gateway import NotificationGateway
from bioflow_ai.services.workflow.dispatcher import WorkflowDispatcher

meta = metadata('bioflow-ai')


@asynccontextmanager
async def lifespan(main_app: FastAPI):  # noqa: ANN201, D103
    with logger.catch():
        create_db_and_tables()
    async with open_checkpointer(setup=True):
        pass
    init_bucket()
    await wait_for_cache_redis()
    main_app.state.redis = await setup_cache_redis()
    notification_gateway = NotificationGateway(notification_redis_client)
    await notification_gateway.start()
    main_app.state.notification_gateway = notification_gateway
    dispatcher = WorkflowDispatcher()
    dispatcher.start()

    yield

    dispatcher.stop()
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
    app.add_middleware(AuthCookieCleanupMiddleware)
    if log_requests:
        app.add_middleware(LogRequestMiddleware)

    app.include_router(auth.router, prefix='/api/v1/auth', tags=['auth'])
    protected = [Depends(get_current_active_user)]
    member_protected = [Depends(require_role(UserRole.MEMBER))]
    app.include_router(
        agent.session_router, prefix='/api/v1/agent-sessions', tags=['agent'], dependencies=member_protected
    )
    app.include_router(agent.run_router, prefix='/api/v1/agent-runs', tags=['agent'], dependencies=member_protected)
    app.include_router(user.router, prefix='/api/v1/user', tags=['user'], dependencies=protected)
    app.include_router(setting.router, prefix='/api/v1/settings', tags=['setting'], dependencies=protected)
    app.include_router(
        notification.router, prefix='/api/v1/notifications', tags=['notification'], dependencies=protected
    )
    app.include_router(skill.router, prefix='/api/v1/skills', tags=['skill'], dependencies=protected)
    app.include_router(project.tag_router, prefix='/api/v1/project-tags', tags=['project'], dependencies=protected)
    app.include_router(
        project.router,
        prefix='/api/v1/projects',
        tags=['project'],
        dependencies=[*protected, Depends(authorize_project_request)],
    )
    app.include_router(
        workflow.router,
        prefix='/api/v1/workflows',
        tags=['workflow'],
        dependencies=[*protected, Depends(authorize_workflow_request)],
    )
    app.include_router(tool.router, prefix='/api/v1/tools', tags=['tool'], dependencies=protected)
    app.include_router(tool_memory.router, prefix='/api/v1/tool-memories', tags=['tool-memory'], dependencies=protected)
    app.include_router(tool.tag_router, prefix='/api/v1/tool-tags', tags=['tool'], dependencies=protected)
    app.include_router(tool.group_router, prefix='/api/v1/tool-groups', tags=['tool'], dependencies=protected)
    app.include_router(tool.image_router, prefix='/api/v1/images', tags=['tool'], dependencies=protected)
    app.include_router(tool.doc_router, prefix='/api/v1/documents', tags=['tool'], dependencies=protected)
    app.include_router(code.router, prefix='/api/v1/codes', tags=['code'], dependencies=protected)
    app.include_router(code_agent.router, prefix='/api/v1/code-agent', tags=['code-agent'], dependencies=protected)
    app.include_router(resource.db_router, prefix='/api/v1/bio_dbs', tags=['resource'], dependencies=protected)
    app.include_router(genome.reference_router, prefix='/api/v1/references', tags=['genome'], dependencies=protected)
    app.include_router(resource.sample_router, prefix='/api/v1/samples', tags=['resource'], dependencies=protected)
    app.include_router(workflow_v2.workflow_router, prefix='/api/v2/workflows', tags=['workflow'])
    app.include_router(workflow_v2.project_router, prefix='/api/v2/projects', tags=['workflow'])
    app.include_router(workflow_v2.run_router, prefix='/api/v2/runs', tags=['workflow'])
    app.include_router(workflow_v2.node_router, prefix='/api/v2/node-runs', tags=['workflow'])

    @app.get('/health', tags=['health'])
    async def health() -> dict[str, str]:
        """Return liveness status once FastAPI startup completes."""
        return {'status': 'ok'}

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
