"""Workflow 模块

工作流处理和任务节点处理器
"""

# 先导入基础框架（注册表、装饰器、基类）
from app.utils.workflow.handler_base import (  # noqa: I001
    BaseTaskHandler,
    TaskContext,
    get_all_handlers,
    get_handler,
    register_handler,
)

# 再导入具体处理器实现（会自动注册）
from app.utils.workflow import handlers  # noqa: F401

# 最后导入核心逻辑
from app.utils.workflow.manager import create_runs_from_samples
from app.utils.workflow.scheduler import (
    build_task_list,
    send_task,
    update_node_id,
)
from app.utils.workflow.stats import (
    get_run_stats_from_redis,
    get_run_stats_from_redis_sync,
)

__all__ = [
    # 基础框架
    'BaseTaskHandler',
    'TaskContext',
    'register_handler',
    'get_handler',
    'get_all_handlers',
    # 核心功能
    'update_node_id',
    'build_task_list',
    'send_task',
    'create_runs_from_samples',
    'get_run_stats_from_redis',
    'get_run_stats_from_redis_sync',
]
