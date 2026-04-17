from collections.abc import Generator
from typing import Annotated, TypeAlias

from fastapi import Depends, Request
from langgraph.checkpoint.redis import RedisSaver
from loguru import logger
from redis.asyncio import Redis
from sqlalchemy import MetaData, create_engine
from sqlmodel import Session, SQLModel

from app.core.config import get_settings

settings = get_settings().persistent

# 配置数据库连接池以支持并发任务
engine = create_engine(
    settings.database.DB_URL,
    pool_size=20,  # 连接池大小
    max_overflow=40,  # 最大溢出连接数
    pool_pre_ping=True,  # 连接前检查可用性
    pool_recycle=3600,  # 连接回收时间
    echo=False,  # 不打印 SQL 语句
)


@logger.catch
def create_db_and_tables() -> None:
    """初始化数据库."""
    SQLModel.metadata.create_all(engine)

    with RedisSaver.from_conn_string(settings.cache.LLM_CACHE) as checkpoint:
        checkpoint.setup()


def drop_table(table_name: str) -> None:
    """删表."""
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table = metadata.tables[table_name]
    if table is not None:
        SQLModel.metadata.drop_all(engine, [table])


def _get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


SessionDep: TypeAlias = Annotated[Session, Depends(_get_session)]  # noqa: UP040


async def get_redis(request: Request) -> Redis:
    """全局请求缓存."""
    return request.app.state.redis


RedisDep: TypeAlias = Annotated[Redis, Depends(get_redis)]  # noqa: UP040
