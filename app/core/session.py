from collections.abc import Generator
from typing import Annotated, TypeAlias

from fastapi import Depends
from loguru import logger
from sqlalchemy import MetaData, create_engine
from sqlmodel import Session, SQLModel

from app.core.config import get_settings
from app.models import Base
from app.models.project import Project
from app.models.resource import Sample
from app.models.user import User

settings = get_settings().persistent


engine = create_engine(
    settings.database.DB_URL.unicode_string(),
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
)


@logger.catch
def create_db_and_tables() -> None:
    """初始化数据库."""
    # Native workflow tables reference these legacy SQLModel tables. Create only
    # the prerequisite parents first, then all native tables, then the remaining
    # SQLModel registry.
    SQLModel.metadata.create_all(
        engine,
        # pyrefly: ignore [missing-attribute]
        tables=[User.__table__, Project.__table__, Sample.__table__],
    )
    Base.metadata.create_all(engine)
    SQLModel.metadata.create_all(engine)


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


SessionDep: TypeAlias = Annotated[Session, Depends(_get_session)]
