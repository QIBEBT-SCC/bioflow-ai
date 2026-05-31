from datetime import UTC, datetime, timedelta
from decimal import Decimal

from fastapi import HTTPException
from loguru import logger
from sqlalchemy import func
from sqlalchemy import select as sa_select
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, col, select
from starlette.status import HTTP_404_NOT_FOUND

from app.models.setting import LLMStatistic
from app.models.user import User
from app.models.workflow import RunInstance
from app.schemas.user import UserDetail, UserUpdate


class UserExistError(ValueError):
    pass


def insert(session: Session, user: User) -> None:
    """Insert a new user into the database."""
    try:
        logger.debug(f'新增用户请求 ({User.model_validate(user)})')
        session.add(user)
        session.commit()
    except IntegrityError as e:
        if 'UNIQUE' in e.args[0]:
            raise UserExistError from e
        raise


def get(session: Session, email: str) -> User | None:
    """Return a user by email address."""
    stmt = select(User).where(User.email == email)
    return session.exec(stmt).one_or_none()


def get_by_id(session: Session, user_id: int) -> User:
    """Return a user by primary key."""
    stmt = select(User).where(User.id == user_id)
    user = session.exec(stmt).one_or_none()
    if user is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail='User not found')
    return user


def update(session: Session, user_id: int, data: UserUpdate) -> User:
    """Apply field updates to an existing user and commit."""
    user_db = get_by_id(session, user_id)
    update_data = data.model_dump(exclude_unset=True)
    user_db.sqlmodel_update(update_data)

    session.add(user_db)
    session.commit()
    session.refresh(user_db)
    return user_db


def get_all_with_stats(session: Session) -> list[UserDetail]:
    """Return all users with aggregated LLM cost and run-instance count."""
    one_month_ago = datetime.now(UTC) - timedelta(days=30)

    total_cost_sq = (
        sa_select(col(LLMStatistic.user_id), func.sum(LLMStatistic.price).label('total'))
        .group_by(col(LLMStatistic.user_id))
        .subquery()
    )
    monthly_cost_sq = (
        sa_select(col(LLMStatistic.user_id), func.sum(LLMStatistic.price).label('monthly'))
        .where(col(LLMStatistic.time) >= one_month_ago)
        .group_by(col(LLMStatistic.user_id))
        .subquery()
    )
    run_count_sq = (
        sa_select(col(RunInstance.owner_id), func.count().label('run_count'))
        .group_by(col(RunInstance.owner_id))
        .subquery()
    )

    stmt = (
        sa_select(
            User,
            func.coalesce(total_cost_sq.c.total, 0).label('total_cost'),
            func.coalesce(monthly_cost_sq.c.monthly, 0).label('monthly_cost'),
            func.coalesce(run_count_sq.c.run_count, 0).label('run_count'),
        )
        .outerjoin(total_cost_sq, col(User.id) == total_cost_sq.c.user_id)
        .outerjoin(monthly_cost_sq, col(User.id) == monthly_cost_sq.c.user_id)
        .outerjoin(run_count_sq, col(User.id) == run_count_sq.c.owner_id)
    )

    return [
        UserDetail(
            id=user.id,
            username=user.username,
            email=user.email,
            role=user.role,
            is_active=user.is_active,
            total_cost=Decimal(str(total_cost)),
            monthly_cost=Decimal(str(monthly_cost)),
            run_count=run_count,
        )
        # pyrefly: ignore [no-matching-overload]
        for user, total_cost, monthly_cost, run_count in session.exec(stmt).all()
    ]
