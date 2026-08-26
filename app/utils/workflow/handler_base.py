"""任务处理器基础框架.

提供任务处理器的基类、装饰器和注册表机制
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from celery import Signature
from loguru import logger
from sqlalchemy.orm import Session

from app.models.workflow import TaskInstance
from app.schemas.react_flow import TaskNode
from app.schemas.user import CurrentUser
from app.services import workflow_runtime


@dataclass
class TaskContext:
    """任务处理上下文, 封装处理器需要的所有共享数据."""

    session: Session
    project_id: int | None
    sample_uid: uuid.UUID | None
    sample_name: str | None
    owner: CurrentUser

    output_base: str
    run_uid: uuid.UUID
    task_index: int

    downstream_files: list[str] = field(default_factory=list)


# ==================== 任务处理器注册表 ====================

# 全局注册表: 映射节点类型到处理器实例
_TASK_HANDLERS: dict[str, 'BaseTaskHandler'] = {}


class EmptyNodeTypeError(ValueError):
    pass


class NotRegisterError(Exception):
    pass


def register_handler(*node_types: str):  # noqa: ANN201
    """装饰器: 注册任务处理器.

    支持单个或多个节点类型注册到同一个处理器

    Usage:
        # 单个类型
        @register_handler('tool')
        class ToolTaskHandler(BaseTaskHandler):
            ...

        # 多个类型 (别名)
        @register_handler('type1', 'type2', 'type3')
        class MultiTypeHandler(BaseTaskHandler):
            ...
    """
    if not node_types:
        err_msg = '至少需要提供一个节点类型'
        raise EmptyNodeTypeError(err_msg)

    def decorator(handler_class: type['BaseTaskHandler']) -> type['BaseTaskHandler']:
        # 创建一个处理器实例
        handler_instance = handler_class()

        # 为所有指定的类型注册同一个实例
        for node_type in node_types:
            _TASK_HANDLERS[node_type] = handler_instance
            logger.debug(f'已注册任务处理器: {node_type} -> {handler_class.__name__}')

        return handler_class

    return decorator


def get_handler(node_type: str) -> Optional['BaseTaskHandler']:
    """获取指定节点类型的处理器."""
    return _TASK_HANDLERS.get(node_type)


def get_all_handlers() -> dict[str, 'BaseTaskHandler']:
    """获取所有已注册的处理器."""
    return _TASK_HANDLERS.copy()


def register_alias(alias: str, original_type: str) -> None:
    """为已注册的处理器添加别名.

    Args:
        alias: 新的别名
        original_type: 原始类型名称

    Raises:
        ValueError: 如果原始类型未注册

    Usage:
        @register_handler('primary_name')
        class MyHandler(BaseTaskHandler):
            ...

        # 添加别名
        register_alias('alias1', 'primary_name')
        register_alias('alias2', 'primary_name')
    """
    if original_type not in _TASK_HANDLERS:
        err_msg = f'Original type {original_type} not registered'
        raise NotRegisterError(err_msg)

    handler = _TASK_HANDLERS[original_type]
    _TASK_HANDLERS[alias] = handler
    logger.debug(f'已注册别名: {alias} -> {handler.__class__.__name__} (原始类型: {original_type})')


# ==================== 任务处理器基类 ====================


class BaseTaskHandler(ABC):
    """任务处理器基类."""

    track_progress = True

    @abstractmethod
    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:
        """创建 TaskInstance 数据库记录."""

    @abstractmethod
    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Optional[Signature]:
        """创建 Celery 任务签名."""

    def handle(self, task_node: TaskNode, context: TaskContext) -> tuple[Optional[Signature], int]:
        """处理任务节点.

        Returns:
            tuple: (celery_task, new_task_index)
        """
        # 创建数据库记录
        task_instance = self.create_task_instance(task_node, context)
        task_instance.track_progress = self.track_progress
        workflow_runtime.create_task(context.session, task_instance)

        # 创建 Celery 任务
        celery_task = self.create_celery_task(task_node, context)

        # 返回任务和更新后的索引 (默认不递增)
        return celery_task, context.task_index
