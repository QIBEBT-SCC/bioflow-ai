from datetime import datetime
from decimal import Decimal

from sqlalchemy import func
from sqlalchemy import select as sa_select
from sqlmodel import Session, col, select
from sqlmodel.sql.expression import Select

from app.models.setting import LLMModel, LLMStatistic
from app.models.user import User


def insert(session: Session, statistic: LLMStatistic) -> None:
    """Insert a new LLM usage statistic record into the database."""
    session.add(statistic)
    session.commit()


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
    def apply_date_filter(stmt: Select) -> Select:
        if start_date is not None:
            stmt = stmt.where(col(LLMStatistic.time) >= start_date)
        if end_date is not None:
            stmt = stmt.where(col(LLMStatistic.time) <= end_date)
        return stmt

    # 总体统计
    total_statement = select(
        func.sum(LLMStatistic.input_tokens).label('total_input_tokens'),
        func.sum(LLMStatistic.output_tokens).label('total_output_tokens'),
        func.sum(LLMStatistic.cache_read).label('total_cache_read'),
        func.sum(LLMStatistic.price).label('total_price'),
    )
    total_statement = apply_date_filter(total_statement)
    total_result = session.exec(total_statement).one()

    # 按 agent 分组统计
    by_agent_statement = sa_select(
        col(LLMStatistic.agent_name),
        func.count(col(LLMStatistic.id)).label('count'),
        func.sum(LLMStatistic.input_tokens).label('total_input_tokens'),
        func.sum(LLMStatistic.output_tokens).label('total_output_tokens'),
        func.sum(LLMStatistic.cache_read).label('total_cache_read'),
        func.sum(LLMStatistic.price).label('total_price'),
    ).group_by(LLMStatistic.agent_name)
    # pyrefly: ignore [bad-argument-type]
    by_agent_statement = apply_date_filter(by_agent_statement)
    by_agent_results = session.exec(by_agent_statement).all()

    # 按 model 分组统计 (JOIN 获取 model name)
    by_model_statement = (
        sa_select(
            col(LLMModel.name).label('model_name'),
            func.count(col(LLMStatistic.id)).label('count'),
            func.sum(LLMStatistic.input_tokens).label('total_input_tokens'),
            func.sum(LLMStatistic.output_tokens).label('total_output_tokens'),
            func.sum(LLMStatistic.cache_read).label('total_cache_read'),
            func.sum(LLMStatistic.price).label('total_price'),
        )
        .join(LLMModel, col(LLMStatistic.model_id) == LLMModel.id, isouter=True)
        .group_by(LLMModel.name)
    )
    # pyrefly: ignore [bad-argument-type]
    by_model_statement = apply_date_filter(by_model_statement)
    by_model_results = session.exec(by_model_statement).all()

    # 按 setting_key (类型) 分组统计
    by_type_statement = (
        sa_select(
            col(LLMStatistic.setting_key),
            func.count(col(LLMStatistic.id)).label('count'),
            func.sum(LLMStatistic.input_tokens).label('total_input_tokens'),
            func.sum(LLMStatistic.output_tokens).label('total_output_tokens'),
            func.sum(LLMStatistic.cache_read).label('total_cache_read'),
            func.sum(LLMStatistic.price).label('total_price'),
        )
        .where(col(LLMStatistic.setting_key).is_not(None))
        .group_by(LLMStatistic.setting_key)
    )
    # pyrefly: ignore [bad-argument-type]
    by_type_statement = apply_date_filter(by_type_statement)
    by_type_results = session.exec(by_type_statement).all()

    # 按 user 分组统计
    by_user_statement = (
        sa_select(
            col(User.username).label('user_name'),
            func.count(col(LLMStatistic.id)).label('count'),
            func.sum(col(LLMStatistic.input_tokens)).label('total_input_tokens'),
            func.sum(col(LLMStatistic.output_tokens)).label('total_output_tokens'),
            func.sum(col(LLMStatistic.cache_read)).label('total_cache_read'),
            func.sum(col(LLMStatistic.price)).label('total_price'),
        )
        .join(User, col(LLMStatistic.user_id) == User.id, isouter=True)
        .group_by(User.username)
    )
    # pyrefly: ignore [bad-argument-type]
    by_user_statement = apply_date_filter(by_user_statement)
    by_user_results = session.exec(by_user_statement).all()

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
                'model_name': row.model_name or 'Unknown',
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
    # 构建查询, JOIN 获取 model name 和 user name
    statement = sa_select(
        col(LLMStatistic.id),
        col(LLMStatistic.agent_name),
        col(LLMModel.name).label('model_name'),
        col(LLMStatistic.setting_key),
        col(LLMStatistic.input_tokens),
        col(LLMStatistic.output_tokens),
        col(LLMStatistic.cache_read),
        col(LLMStatistic.price),
        col(LLMStatistic.time),
    ).join(LLMModel, col(LLMStatistic.model_id) == LLMModel.id, isouter=True)

    if start_date is not None:
        statement = statement.where(col(LLMStatistic.time) >= start_date)

    if end_date is not None:
        statement = statement.where(col(LLMStatistic.time) <= end_date)

    # 获取总数
    count_statement = select(func.count(col(LLMStatistic.id)))
    if start_date is not None:
        count_statement = count_statement.where(col(LLMStatistic.time) >= start_date)
    if end_date is not None:
        count_statement = count_statement.where(col(LLMStatistic.time) <= end_date)

    total = session.exec(count_statement).one()

    # 查询详细记录 (带分页)
    statement = statement.order_by(col(LLMStatistic.time).desc()).limit(limit).offset(offset)
    # pyrefly: ignore [no-matching-overload]
    results = session.exec(statement).all()

    return {
        'total': total,
        'items': [
            {
                'id': row.id,
                'agent_name': row.agent_name,
                'model_name': row.model_name,
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
    """刷新对应记录的thread_id."""
    statement = select(LLMStatistic).where(LLMStatistic.thread == thread_id)
    results = session.exec(statement).all()
    for record in results:
        record.thread = new_id
    session.commit()
