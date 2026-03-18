"""任务处理器实现.

包含所有具体的任务节点处理器
"""

import uuid
from typing import Optional

from celery import Signature

import app.crud.resource.bio_db as db_crud
import app.crud.resource.sample as sample_crud
import app.crud.tools.tool as tool_crud
from app.core.config import get_settings
from app.models.resource import SampleFileType
from app.models.workflow import Status, TaskInstance
from app.schemas.react_flow import TaskNode
from app.tasks.load_tasks.load_file import load_file_task, load_sequence_task
from app.tasks.load_tasks.load_genome import load_hg38_task, load_mm39_task, load_ncbi_genome_task
from app.tasks.load_tasks.static_value import set_value_task
from app.tasks.package_result import package_task
from app.tasks.run_tool import docker_task
from app.utils.workflow.handler_base import BaseTaskHandler, TaskContext, register_handler


class EmptyToolUIDError(ValueError):
    pass


class MissingReadError(ValueError):
    pass


@register_handler('tool')
class ToolTaskHandler(BaseTaskHandler):
    """工具任务处理器."""

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        if tool_uid := task_node.data.get('tool_uid'):
            tool_data = tool_crud.get(context.session, tool_uid)

            return TaskInstance(
                uid=uuid.UUID(task_node.id),
                instance_uid=context.run_uid,
                tool_uid=uuid.UUID(tool_uid),
                owner_id=context.owner.id,
                name=tool_data.name,
                status=Status.WAITING,
            )

        raise EmptyToolUIDError

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Signature:  # noqa: D102
        if tool_uid := task_node.data.get('tool_uid'):
            tool_data = tool_crud.get(context.session, tool_uid)

            output_dir = (
                f'{get_settings().persistent.STORAGE_PATH}/'
                f'{context.owner.username}/'
                f'{context.output_base}/'
                f'{context.task_index}_{tool_data.name}'
            )

            return docker_task.s(
                tool_uid=tool_uid,
                inputs=task_node.kwarg,
                output_dir=output_dir,
                params=task_node.data.get('args'),
                sample_name=context.sample_name,
            ).set(task_id=task_node.id)

        raise EmptyToolUIDError

    def handle(self, task_node: TaskNode, context: TaskContext) -> tuple[Optional[Signature], int]:
        """工具任务需要递增 task_index."""
        celery_task, _ = super().handle(task_node, context)
        return celery_task, context.task_index + 1


@register_handler('resource_db')
class ResourceDbTaskHandler(BaseTaskHandler):
    """数据库资源任务处理器."""

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        return TaskInstance(
            uid=uuid.UUID(task_node.id),
            instance_uid=context.run_uid,
            owner_id=context.owner.id,
            name='read_file',
            status=Status.WAITING,
        )

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Signature:  # noqa: D102
        db_path = db_crud.get(context.session, int(task_node.data.get('db_id'))).path
        return load_file_task.s(file_path=db_path).set(task_id=task_node.id)


@register_handler('resource_sequence')
class ResourceSequenceTaskHandler(BaseTaskHandler):
    """测序数据资源任务处理器."""

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        return TaskInstance(
            uid=uuid.UUID(task_node.id),
            instance_uid=context.run_uid,
            owner_id=context.owner.id,
            name='read_file',
            status=Status.WAITING,
        )

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Signature:  # noqa: D102
        if context.auto_fill:
            sample_uid = context.auto_fill.split(':')[1]
            sample = sample_crud.get_with_files(context.session, sample_uid)  # type: ignore

            r1_path = next(
                (f.file_path for f in sample.sample_files if f.data_type == SampleFileType.SEQUENCING_R1),
                None,
            )
            r2_path = next(
                (f.file_path for f in sample.sample_files if f.data_type == SampleFileType.SEQUENCING_R2),
                None,
            )

            if r1_path is None:
                err_msg = 'Sample missing r1 sequencing file'
                raise MissingReadError(err_msg)
            if r2_path is None:
                err_msg = 'Sample missing r2 sequencing file'
                raise MissingReadError(err_msg)

            return load_sequence_task.s(args={'r1': r1_path, 'r2': r2_path}).set(task_id=task_node.id)

        return load_sequence_task.s(args=task_node.data).set(task_id=task_node.id)


@register_handler('resource_genome')
class ResourceGenomeTaskHandler(BaseTaskHandler):
    """参考基因组资源任务处理器."""

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        return TaskInstance(
            uid=uuid.UUID(task_node.id),
            instance_uid=context.run_uid,
            owner_id=context.owner.id,
            name='load_reference_genome',
            status=Status.WAITING,
        )

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Signature:  # noqa: D102
        return load_ncbi_genome_task.s(
            inputs=task_node.kwarg,
            required_index=task_node.data.get('required_index', []),
        ).set(task_id=task_node.id)


@register_handler('resource_GRCh38')
class ResourceHG38TaskHandler(BaseTaskHandler):
    """人类基因组资源任务处理器."""

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        return TaskInstance(
            uid=uuid.UUID(task_node.id),
            instance_uid=context.run_uid,
            owner_id=context.owner.id,
            name='load_reference_genome',
            status=Status.WAITING,
        )

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Signature:  # noqa: D102
        return load_hg38_task.s(
            required_index=task_node.data.get('required_index', []),
        ).set(task_id=task_node.id)


@register_handler('resource_GRCm39')
class ResourceMM39TaskHandler(BaseTaskHandler):
    """小鼠基因组资源任务处理器."""

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        return TaskInstance(
            uid=uuid.UUID(task_node.id),
            instance_uid=context.run_uid,
            owner_id=context.owner.id,
            name='load_reference_genome',
            status=Status.WAITING,
        )

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Signature:  # noqa: D102
        return load_mm39_task.s(
            required_index=task_node.data.get('required_index', []),
        ).set(task_id=task_node.id)


@register_handler('resource_file')
class ResourceFileTaskHandler(BaseTaskHandler):
    """文件资源任务处理器."""

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        return TaskInstance(
            uid=uuid.UUID(task_node.id),
            instance_uid=context.run_uid,
            owner_id=context.owner.id,
            name='read_file',
            status=Status.WAITING,
        )

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Signature:  # noqa: D102
        return load_file_task.s(file_path=task_node.data.get('file_path')).set(task_id=task_node.id)


@register_handler('value_string')
class ValueStringTaskHandler(BaseTaskHandler):
    """字符串值任务处理器."""

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        return TaskInstance(
            uid=uuid.UUID(task_node.id),
            instance_uid=context.run_uid,
            owner_id=context.owner.id,
            name='value',
            status=Status.WAITING,
        )

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Signature:  # noqa: D102
        return set_value_task.s(value=task_node.data.get('value', ''), value_type='string').set(task_id=task_node.id)


@register_handler('process_copy2folder')
class ProcessCopy2FolderTaskHandler(BaseTaskHandler):
    """复制到文件夹任务处理器(占位)."""

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        # 暂时创建一个空的 TaskInstance
        return TaskInstance(
            uid=uuid.UUID(task_node.id),
            instance_uid=context.run_uid,
            owner_id=context.owner.id,
            name='copy_to_folder',
            status=Status.WAITING,
        )

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Optional[Signature]:  # noqa: D102
        # 当前未实现, 返回 None
        return None

    def handle(self, task_node: TaskNode, context: TaskContext) -> tuple[Optional[Signature], int]:
        """Copy2Folder 目前不创建任务."""
        return None, context.task_index


@register_handler('code_python', 'code_R')
class SimpleCodeTaskHandler(BaseTaskHandler):
    """处理下游分析代码节点.

    这个处理器不创建 Celery 任务和 TaskInstance
    只将代码信息添加到 context.downstream_files 中供后续使用
    """

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        pass

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Optional[Signature]:  # noqa: D102
        pass

    def handle(self, task_node: TaskNode, context: TaskContext) -> tuple[Optional[Signature], int]:
        """不创建任务, 只记录下游代码信息."""
        node_inputs = task_node.kwarg['input_files']
        input_files = node_inputs if isinstance(node_inputs, list) else [node_inputs]

        context.downstream_files.extend(input_files)

        return None, context.task_index


@register_handler('summary')
class SummaryHandler(BaseTaskHandler):
    """处理下游分析代码节点.

    这个处理器不创建 Celery 任务和 TaskInstance
    只将代码信息添加到 context.downstream_files 中供后续使用
    """

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        return TaskInstance(
            uid=context.run_uid,  # 最终打包节点的任务uuid与运行实例的uuid一致
            instance_uid=context.run_uid,
            owner_id=context.owner.id,
            name='package',
            status=Status.WAITING,
        )

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Optional[Signature]:  # noqa: D102
        if context.downstream_files:
            downstream_files = list(set(context.downstream_files))
            output_dir = (
                f'{get_settings().persistent.STORAGE_PATH}/{context.owner.username}/{context.output_base}/summary'
            )

            return package_task.s(
                run_uid=f'{context.run_uid}',
                inputs=task_node.kwarg,
                output_dir=output_dir,
                sample_name=context.sample_name,
            ).set(task_id=context.run_uid)  # 最终打包节点的任务uuid与运行实例的uuid一致

        return None
