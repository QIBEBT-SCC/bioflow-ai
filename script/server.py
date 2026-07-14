from contextlib import AsyncExitStack, asynccontextmanager
from importlib.metadata import metadata

import uvicorn
from fastapi import FastAPI
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
from loguru import logger
from starlette.middleware.cors import CORSMiddleware

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
from app.utils.middleware import LogRequestMiddleware
from llm.prebuild.dispatch import build_agents

meta = metadata('bioflow-ai')


@asynccontextmanager
async def lifespan(main_app: FastAPI):  # noqa: ANN201, D103
    create_db_and_tables()
    init_bucket()
    main_app.state.redis = await setup_cache_redis()

    async with AsyncExitStack() as stack:
        # A single long-lived checkpointer shared by every top-level agent. The agents are
        # compiled once here and cached on app.state; per request only thread_id/context vary.
        checkpointer = await stack.enter_async_context(
            AsyncRedisSaver.from_conn_string(get_settings().persistent.cache.LLM_CACHE.unicode_string())
        )
        checkpointer.serde = JsonPlusRedisSerializer(
            allowed_msgpack_modules=[
                ('llm.states', 'BuildEdge'),
                ('llm.states', 'BuildNode'),
                ('llm.utils.middleware.todo', 'Todo'),
                ('llm.utils.uid_mapper', 'BidirectionalMapper'),
                ('app.models.tools', 'FileMount'),
                ('app.models.tools', 'ImageConfig'),
                ('app.models.tools', 'ParamDefine'),
            ]
        )
        main_app.state.checkpointer = checkpointer
        main_app.state.chat_agents = build_agents(checkpointer)

        yield

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
