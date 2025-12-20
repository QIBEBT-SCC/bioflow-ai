from typing import Annotated, TypeAlias

from fastapi import Depends
from langgraph.checkpoint.redis import RedisSaver
from loguru import logger
from sqlalchemy import MetaData, create_engine
from sqlmodel import Session, SQLModel

from app.core.config import get_settings

engine = create_engine(get_settings().persistent.DB_URL)


@logger.catch
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

    with RedisSaver.from_conn_string(get_settings().persistent.LLM_CACHE) as checkpoint:
        checkpoint.setup()


def drop_table(table_name: str):
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table = metadata.tables[table_name]
    if table is not None:
        SQLModel.metadata.drop_all(engine, [table])


def get_session():
    with Session(engine) as session:
        yield session


SessionDep: TypeAlias = Annotated[Session, Depends(get_session)]
