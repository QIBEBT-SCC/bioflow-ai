"""路径解析工具

提供 proj: 和 sample: 引用的解析功能
"""

import uuid

from sqlmodel import Session

import app.crud.project as project_crud
import app.crud.resource.sample as sample_crud
from app.utils.workflow.handler_base import TaskContext


def resolve_path(
    path_or_reference: str,
    context: TaskContext,
    session: Session,
    project_id: int,
) -> str:
    """解析路径引用为实际文件路径

    Args:
        path_or_reference: 普通路径或引用（如 "proj:index" 或 "sample:r1"）
        context: 任务上下文，包含 auto_fill 信息
        session: 数据库会话
        project_id: 项目 ID，用于项目级查找

    Returns:
        解析后的文件路径

    Raises:
        ValueError: 如果引用无法解析（在任务执行时抛出）

    Note:
        错误在运行时任务执行时抛出，而非工作流验证时。
        这允许在某些映射尚未配置时创建工作流。
    """
    if not path_or_reference.startswith(('proj:', 'sample:')):
        # 普通路径，直接返回
        return path_or_reference

    if path_or_reference.startswith('proj:'):
        return _resolve_project_path(path_or_reference, session, project_id)

    if path_or_reference.startswith('sample:'):
        return _resolve_sample_path(path_or_reference, context, session)

    raise ValueError(f'Invalid path reference: {path_or_reference}')


def _resolve_project_path(reference: str, session: Session, project_id: int) -> str:
    """解析 proj:<keyword> 为项目文件映射中的文件路径"""
    keyword = reference.split(':', 1)[1]

    # 查询 project_file_mappings 表
    mapping = project_crud.get_file_mapping(session, project_id, keyword)

    if not mapping:
        raise ValueError(f'Project file mapping not found: {keyword}')

    return mapping.file_path


def _resolve_sample_path(reference: str, context: TaskContext, session: Session) -> str:
    """解析 sample:<tag> 为样本文件路径

    仅使用精确标签匹配 - 不支持通配符或模式
    """
    if not context.auto_fill or not context.auto_fill.startswith('sample:'):
        raise ValueError('Sample path resolution requires auto_fill context')

    tag = reference.split(':', 1)[1]
    sample_uid_str = context.auto_fill.split(':', 1)[1]
    sample_uid = uuid.UUID(sample_uid_str)

    sample = sample_crud.get_with_files(session, sample_uid)
    if not sample:
        raise ValueError(f'Sample not found: {sample_uid}')

    # 查找具有精确匹配标签的文件（无通配符）
    matching_file = next((f for f in sample.sample_files if f.tag == tag), None)

    if not matching_file:
        raise ValueError(f"Sample file with tag '{tag}' not found")

    return matching_file.file_path
