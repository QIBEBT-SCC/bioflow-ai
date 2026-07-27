from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.models.setting import LLMModel, LLMStatistic
from app.models.user import User


def insert(session: Session, statistic: LLMStatistic) -> None:
    """Stage a new LLM usage statistic record."""
    session.add(statistic)
    session.flush()


def get_summary(
    session: Session,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict:
    """获取 LLM 使用统计概览.

    Args:
        session: 数据库会话
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        包含总体统计和各维度分组统计的字典
    """

    # 构建基础条件
    def apply_date_filter(stmt: Select[Any]) -> Select[Any]:
        if start_date is not None:
            stmt = stmt.where(LLMStatistic.time >= start_date)
        if end_date is not None:
            stmt = stmt.where(LLMStatistic.time <= end_date)
        return stmt

    # 总体统计
    total_statement = select(
        func.sum(LLMStatistic.input_tokens).label('total_input_tokens'),
        func.sum(LLMStatistic.output_tokens).label('total_output_tokens'),
        func.sum(LLMStatistic.cache_read).label('total_cache_read'),
        func.sum(LLMStatistic.price).label('total_price'),
    )
    total_statement = apply_date_filter(total_statement)
    total_result = session.execute(total_statement).one()

    # 按 agent 分组统计
    by_agent_statement = select(
        LLMStatistic.agent_name,
        func.count(LLMStatistic.id).label('count'),
        func.sum(LLMStatistic.input_tokens).label('total_input_tokens'),
        func.sum(LLMStatistic.output_tokens).label('total_output_tokens'),
        func.sum(LLMStatistic.cache_read).label('total_cache_read'),
        func.sum(LLMStatistic.price).label('total_price'),
    ).group_by(LLMStatistic.agent_name)
    by_agent_statement = apply_date_filter(by_agent_statement)
    by_agent_results = session.execute(by_agent_statement).all()

    # 按 model 分组统计 (JOIN 获取模型显示名称和 API 名称)
    by_model_statement = (
        select(
            LLMModel.display_name,
            LLMModel.name,
            func.count(LLMStatistic.id).label('count'),
            func.sum(LLMStatistic.input_tokens).label('total_input_tokens'),
            func.sum(LLMStatistic.output_tokens).label('total_output_tokens'),
            func.sum(LLMStatistic.cache_read).label('total_cache_read'),
            func.sum(LLMStatistic.price).label('total_price'),
        )
        .join(LLMModel, LLMStatistic.model_id == LLMModel.id, isouter=True)
        .group_by(LLMModel.id, LLMModel.display_name, LLMModel.name)
    )
    by_model_statement = apply_date_filter(by_model_statement)
    by_model_results = session.execute(by_model_statement).all()

    # 按 setting_key (类型) 分组统计
    by_type_statement = (
        select(
            LLMStatistic.setting_key,
            func.count(LLMStatistic.id).label('count'),
            func.sum(LLMStatistic.input_tokens).label('total_input_tokens'),
            func.sum(LLMStatistic.output_tokens).label('total_output_tokens'),
            func.sum(LLMStatistic.cache_read).label('total_cache_read'),
            func.sum(LLMStatistic.price).label('total_price'),
        )
        .where(LLMStatistic.setting_key.is_not(None))
        .group_by(LLMStatistic.setting_key)
    )
    by_type_statement = apply_date_filter(by_type_statement)
    by_type_results = session.execute(by_type_statement).all()

    # 按 user 分组统计
    by_user_statement = (
        select(
            User.username.label('user_name'),
            func.count(LLMStatistic.id).label('count'),
            func.sum(LLMStatistic.input_tokens).label('total_input_tokens'),
            func.sum(LLMStatistic.output_tokens).label('total_output_tokens'),
            func.sum(LLMStatistic.cache_read).label('total_cache_read'),
            func.sum(LLMStatistic.price).label('total_price'),
        )
        .join(User, LLMStatistic.user_id == User.id, isouter=True)
        .group_by(User.username)
    )
    by_user_statement = apply_date_filter(by_user_statement)
    by_user_results = session.execute(by_user_statement).all()

    return {
        'total': {
            'total_input_tokens': int(total_result.total_input_tokens or 0),
            'total_output_tokens': int(total_result.total_output_tokens or 0),
            'total_cache_read': int(total_result.total_cache_read or 0),
            'total_price': Decimal(str(total_result.total_price or 0)),
        },
        'by_agent': [
            {
                'agent_name': row.agent_name,
                'count': row.count,
                'total_input_tokens': int(row.total_input_tokens or 0),
                'total_output_tokens': int(row.total_output_tokens or 0),
                'total_cache_read': int(row.total_cache_read or 0),
                'total_price': Decimal(str(row.total_price or 0)),
            }
            for row in by_agent_results
        ],
        'by_model': [
            {
                'display_name': row.display_name or 'Unknown',
                'name': row.name or 'Unknown',
                'count': row.count,
                'total_input_tokens': int(row.total_input_tokens or 0),
                'total_output_tokens': int(row.total_output_tokens or 0),
                'total_cache_read': int(row.total_cache_read or 0),
                'total_price': Decimal(str(row.total_price or 0)),
            }
            for row in by_model_results
        ],
        'by_type': [
            {
                'setting_key': row.setting_key,
                'count': row.count,
                'total_input_tokens': int(row.total_input_tokens or 0),
                'total_output_tokens': int(row.total_output_tokens or 0),
                'total_cache_read': int(row.total_cache_read or 0),
                'total_price': Decimal(str(row.total_price or 0)),
            }
            for row in by_type_results
        ],
        'by_user': [
            {
                'user_name': row.user_name or 'Unknown',
                'count': row.count,
                'total_input_tokens': int(row.total_input_tokens or 0),
                'total_output_tokens': int(row.total_output_tokens or 0),
                'total_cache_read': int(row.total_cache_read or 0),
                'total_price': Decimal(str(row.total_price or 0)),
            }
            for row in by_user_results
        ],
    }


def get_details(
    session: Session,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """获取 LLM 使用详细记录列表.

    Args:
        session: 数据库会话
        start_date: 开始日期, 如果为 None 则不限制开始时间
        end_date: 结束日期, 如果为 None 则不限制结束时间
        limit: 每页记录数
        offset: 偏移量

    Returns:
        详细记录列表和总数
    """
    # 构建查询, JOIN 获取模型显示名称和 API 名称
    statement = select(
        LLMStatistic.id,
        LLMStatistic.agent_name,
        LLMModel.display_name,
        LLMModel.name,
        LLMStatistic.setting_key,
        LLMStatistic.input_tokens,
        LLMStatistic.output_tokens,
        LLMStatistic.cache_read,
        LLMStatistic.price,
        LLMStatistic.time,
    ).join(LLMModel, LLMStatistic.model_id == LLMModel.id, isouter=True)

    if start_date is not None:
        statement = statement.where(LLMStatistic.time >= start_date)

    if end_date is not None:
        statement = statement.where(LLMStatistic.time <= end_date)

    # 获取总数
    count_statement = select(func.count(LLMStatistic.id))
    if start_date is not None:
        count_statement = count_statement.where(LLMStatistic.time >= start_date)
    if end_date is not None:
        count_statement = count_statement.where(LLMStatistic.time <= end_date)

    total = session.scalar(count_statement) or 0

    # 查询详细记录 (带分页)
    statement = statement.order_by(LLMStatistic.time.desc()).limit(limit).offset(offset)
    results = session.execute(statement).all()

    return {
        'total': total,
        'items': [
            {
                'id': row.id,
                'agent_name': row.agent_name,
                'display_name': row.display_name,
                'name': row.name,
                'setting_key': row.setting_key,
                'input_tokens': row.input_tokens,
                'output_tokens': row.output_tokens,
                'cache_read': row.cache_read,
                'price': row.price,
                'time': row.time,
            }
            for row in results
        ],
    }


def refresh_thread_id(session: Session, thread_id: str, new_id: str) -> None:
    """Stage a thread-ID replacement for matching statistic records."""
    statement = select(LLMStatistic).where(LLMStatistic.thread == thread_id)
    results = session.scalars(statement).all()
    for record in results:
        record.thread = new_id
    session.flush()
