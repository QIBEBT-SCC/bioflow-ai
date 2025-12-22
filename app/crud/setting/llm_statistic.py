from datetime import datetime
from typing import Optional

from sqlmodel import Session, col, select

from app.models.setting import LLMStatistic


def insert(session: Session, statistic: LLMStatistic):
    session.add(statistic)
    session.commit()


def get_all_by_date(
    session: Session,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
):
    """
    根据日期范围查询 LLM 统计记录

    Args:
        session: 数据库会话
        start_date: 开始日期（包含），如果为 None 则不限制开始时间
        end_date: 结束日期（包含），如果为 None 则不限制结束时间

    Returns:
        符合条件的 LLMStatistic 记录列表
    """
    statement = select(LLMStatistic)

    if start_date is not None:
        statement = statement.where(LLMStatistic.time >= start_date)

    if end_date is not None:
        statement = statement.where(LLMStatistic.time <= end_date)

    statement = statement.order_by(col(LLMStatistic.time).desc())

    results = session.exec(statement).all()
    return results
