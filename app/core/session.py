from typing import Annotated

from fastapi import Depends
from loguru import logger
from sqlalchemy import MetaData, create_engine
from sqlalchemy_utils import database_exists, create_database
from sqlmodel import Session, SQLModel

from app.core.config import get_settings

if not database_exists(get_settings().persistent.DB_URL):
    create_database(get_settings().persistent.DB_URL)

engine = create_engine(get_settings().persistent.DB_URL)


@logger.catch
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def drop_table(table_name: str):
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table = metadata.tables[table_name]
    if table is not None:
        SQLModel.metadata.drop_all(engine, [table])


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
