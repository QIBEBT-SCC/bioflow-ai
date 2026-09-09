from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.setting import LLMStatistic
from app.models.user import User
from app.models.workflow import WorkflowRun
from app.schemas.user import UserDetail, UserUpdate
from app.schemas.user_config import UserRole


def insert(session: Session, user: User) -> User:
    """Stage a new user and populate database-generated values."""
    session.add(user)
    session.flush()
    session.refresh(user)
    return user


def get(session: Session, email: str) -> User | None:
    """Return a user by email address."""
    return session.scalar(select(User).where(func.lower(User.email) == email.strip().casefold()))


def get_by_id(session: Session, user_id: int) -> User | None:
    """Return a user by primary key."""
    return session.get(User, user_id)


def count_active_admins(session: Session, *, lock: bool = False) -> int:
    """Return the number of active administrator accounts."""
    if lock:
        statement = select(User).where(User.role == UserRole.ADMIN, User.is_active.is_(True)).with_for_update()
        return len(session.scalars(statement).all())
    return (
        session.scalar(
            select(func.count()).select_from(User).where(User.role == UserRole.ADMIN, User.is_active.is_(True))
        )
        or 0
    )


def update(session: Session, user_id: int, data: UserUpdate) -> User | None:
    """Stage field updates for an existing user."""
    user = get_by_id(session, user_id)
    if user is None:
        return None

    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(user, field, value)

    session.flush()
    session.refresh(user)
    return user


def get_all_with_stats(session: Session) -> list[UserDetail]:
    """Return all users with aggregated LLM cost and run-instance count."""
    one_month_ago = datetime.now(UTC) - timedelta(days=30)

    total_cost_sq = (
        select(LLMStatistic.user_id, func.sum(LLMStatistic.price).label('total'))
        .group_by(LLMStatistic.user_id)
        .subquery()
    )
    monthly_cost_sq = (
        select(LLMStatistic.user_id, func.sum(LLMStatistic.price).label('monthly'))
        .where(LLMStatistic.time >= one_month_ago)
        .group_by(LLMStatistic.user_id)
        .subquery()
    )
    run_count_sq = (
        select(WorkflowRun.owner_id, func.count().label('run_count')).group_by(WorkflowRun.owner_id).subquery()
    )

    stmt = (
        select(
            User,
            func.coalesce(total_cost_sq.c.total, 0).label('total_cost'),
            func.coalesce(monthly_cost_sq.c.monthly, 0).label('monthly_cost'),
            func.coalesce(run_count_sq.c.run_count, 0).label('run_count'),
        )
        .outerjoin(total_cost_sq, User.id == total_cost_sq.c.user_id)
        .outerjoin(monthly_cost_sq, User.id == monthly_cost_sq.c.user_id)
        .outerjoin(run_count_sq, User.id == run_count_sq.c.owner_id)
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
        for user, total_cost, monthly_cost, run_count in session.execute(stmt).all()
    ]
