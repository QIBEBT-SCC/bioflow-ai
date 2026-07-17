"""任务处理器实现.

包含所有具体的任务节点处理器
"""

import uuid
from typing import Optional

import nanoid
from celery import Signature

import app.crud.tools.tool as tool_crud
from app.models.workflow import Status, TaskInstance
from app.schemas.react_flow import TaskNode
from app.tasks.code import bash_task
from app.tasks.file_operation import (
    bind_param_task,
    collect_files_task,
    collect_mount_dirs_task,
    create_text_file_task,
    global_mark_task,
    gzip_file_task,
    select_file_task,
)
from app.tasks.load_tasks.load_bio_db import load_bio_db_task
from app.tasks.load_tasks.load_file import load_file_task, load_sequence_task
from app.tasks.load_tasks.load_genome import load_hg38_task, load_mm39_task, load_ncbi_genome_task
from app.tasks.load_tasks.static_value import set_value_task
from app.tasks.run_tool import docker_task
from app.tasks.summary_task.summary import downstream_summary_task, upstream_summary_task
from app.utils.input_format import to_safe_path_segment
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
                base_dir=f'{context.output_base}/{context.task_index}_{to_safe_path_segment(tool_data.name)}',
                status=Status.WAITING,
            )

        raise EmptyToolUIDError

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Signature:  # noqa: D102
        if tool_uid := task_node.data.get('tool_uid'):
            tool_data = tool_crud.get(context.session, tool_uid)

            tool_args = task_node.data.get('modifiable_params')
            return docker_task.s(
                tool_uid=tool_uid,
                inputs=task_node.kwarg,
                output_dir=f'{context.output_base}/{context.task_index}_{to_safe_path_segment(tool_data.name)}',
                params=tool_args,
                sample_name=context.sample_name,
            ).set(task_id=task_node.id)

        raise EmptyToolUIDError

    def handle(self, task_node: TaskNode, context: TaskContext) -> tuple[Optional[Signature], int]:
        """工具任务需要递增 task_index."""
        celery_task, _ = super().handle(task_node, context)
        return celery_task, context.task_index + 1


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
        file_path_ref = task_node.data.get('file_path', '')
        return load_file_task.s(
            file=file_path_ref,
            project_id=context.project_id,
            sample_uid=context.sample_uid,
            sample_name=context.sample_name,
        ).set(task_id=task_node.id)


@register_handler('resource_text_file')
class ResourceTextFileTaskHandler(BaseTaskHandler):
    """内联文本文件资源任务处理器."""

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        output_dir = f'{context.output_base}/{context.task_index}_text_file'
        return TaskInstance(
            uid=uuid.UUID(task_node.id),
            instance_uid=context.run_uid,
            owner_id=context.owner.id,
            name='create_text_file',
            base_dir=output_dir,
            status=Status.WAITING,
        )

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Signature:  # noqa: D102
        output_dir = f'{context.output_base}/{context.task_index}_text_file'
        return create_text_file_task.s(
            file_name=task_node.data.get('file_name', ''),
            content=task_node.data.get('content', ''),
            output_dir=output_dir,
        ).set(task_id=task_node.id)

    def handle(self, task_node: TaskNode, context: TaskContext) -> tuple[Optional[Signature], int]:
        """文本文件使用独立输出目录, 需要递增 task_index."""
        celery_task, _ = super().handle(task_node, context)
        return celery_task, context.task_index + 1


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
        r1_ref = task_node.data.get('r1', 'sample:r1')
        r2_ref = task_node.data.get('r2', 'sample:r2')

        return load_sequence_task.s(
            r1=r1_ref,
            r2=r2_ref,
            project_id=context.project_id,
            sample_uid=context.sample_uid,
            sample_name=context.sample_name,
        ).set(task_id=task_node.id)


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
        return load_bio_db_task.s(db_id=int(task_node.data.get('db_id'))).set(task_id=task_node.id)


@register_handler('resource_ncbi_genome')
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


@register_handler('bind_param')
class BindParamTaskHandler(BaseTaskHandler):
    """绑定参数名任务处理器."""

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        return TaskInstance(
            uid=uuid.UUID(task_node.id),
            instance_uid=context.run_uid,
            owner_id=context.owner.id,
            name='bind_param',
            status=Status.WAITING,
        )

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Signature:  # noqa: D102
        return bind_param_task.s(
            inputs=task_node.kwarg,
            parameter=task_node.data.get('parameter'),
        ).set(task_id=task_node.id)


@register_handler('copy2folder')
class ProcessCopy2FolderTaskHandler(BaseTaskHandler):
    """复制到文件夹任务处理器."""

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        return TaskInstance(
            uid=uuid.UUID(task_node.id),
            instance_uid=context.run_uid,
            owner_id=context.owner.id,
            name='copy2folder',
            status=Status.WAITING,
        )

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Optional[Signature]:  # noqa: D102
        output_dir = f'{context.output_base}/{task_node.data.get("folder_name", nanoid.generate(size=8))}'
        return collect_files_task.s(inputs=task_node.kwarg, output_dir=output_dir).set(task_id=task_node.id)


@register_handler('global_mark')
class GlobalMarkTaskHandler(BaseTaskHandler):
    """项目级文件标记任务处理器."""

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        return TaskInstance(
            uid=uuid.UUID(task_node.id),
            instance_uid=context.run_uid,
            owner_id=context.owner.id,
            name='global_mark',
            status=Status.WAITING,
        )

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Optional[Signature]:  # noqa: D102
        return global_mark_task.s(
            inputs=task_node.kwarg,
            project_id=context.project_id,
            mark_name=task_node.data.get('mark_name', ''),
            description=task_node.data.get('description', ''),
        ).set(task_id=task_node.id)


@register_handler('collect_mount_dirs')
class CollectMountDirsTaskHandler(BaseTaskHandler):
    """收集多个工具目录, 生成可展开挂载的目录清单."""

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        return TaskInstance(
            uid=uuid.UUID(task_node.id),
            instance_uid=context.run_uid,
            owner_id=context.owner.id,
            name='collect_mount_dirs',
            base_dir=f'{context.output_base}/{task_node.id}_mount_dirs',
            status=Status.WAITING,
        )

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Optional[Signature]:  # noqa: D102
        output_dir = f'{context.output_base}/{task_node.id}_mount_dirs'
        return collect_mount_dirs_task.s(inputs=task_node.kwarg, output_dir=output_dir).set(task_id=task_node.id)


@register_handler('select_file_in_folder')
class SelectTaskHandler(BaseTaskHandler):
    """提取文件任务处理器."""

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        return TaskInstance(
            uid=uuid.UUID(task_node.id),
            instance_uid=context.run_uid,
            owner_id=context.owner.id,
            name='select_file',
            status=Status.WAITING,
        )

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Optional[Signature]:  # noqa: D102
        return select_file_task.s(inputs=task_node.kwarg, file_name=task_node.data.get('file_name')).set(
            task_id=task_node.id
        )


@register_handler('gzip')
class GzipTaskHandler(BaseTaskHandler):
    """gzip压缩任务处理器."""

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        return TaskInstance(
            uid=uuid.UUID(task_node.id),
            instance_uid=context.run_uid,
            owner_id=context.owner.id,
            name='gzip',
            status=Status.WAITING,
        )

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Optional[Signature]:  # noqa: D102
        return gzip_file_task.s(inputs=task_node.kwarg).set(task_id=task_node.id)


@register_handler('code_bash')
class BashCodeTaskHandler(BaseTaskHandler):
    """Bash 脚本执行任务处理器."""

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        return TaskInstance(
            uid=uuid.UUID(task_node.id),
            instance_uid=context.run_uid,
            owner_id=context.owner.id,
            name='code_bash',
            status=Status.WAITING,
        )

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Signature:  # noqa: D102
        output_dir = f'{context.output_base}/{context.task_index}_bash'
        return bash_task.s(
            inputs=task_node.kwarg,
            code=task_node.data.get('code', ''),
            output_dir=output_dir,
        ).set(task_id=task_node.id)

    def handle(self, task_node: TaskNode, context: TaskContext) -> tuple[Optional[Signature], int]:
        """Bash 任务产生独立输出目录, 需要递增 task_index."""
        celery_task, _ = super().handle(task_node, context)
        return celery_task, context.task_index + 1


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


@register_handler('downstream_summary')
class DownSummaryHandler(BaseTaskHandler):
    """处理下游分析总结节点."""

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        return TaskInstance(
            uid=uuid.UUID(task_node.id),
            instance_uid=context.run_uid,
            owner_id=context.owner.id,
            name='summary',
            base_dir=f'{context.output_base}/downstream',
            status=Status.WAITING,
        )

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Signature:  # noqa: D102
        return downstream_summary_task.s(
            inputs=task_node.kwarg,
            prompt=task_node.data.get('prompt', ''),
            output_dir=f'{context.output_base}/downstream',
        ).set(task_id=task_node.id)


@register_handler('upstream_summary')
class UpSummaryHandler(BaseTaskHandler):
    """上游分析总结."""

    def create_task_instance(self, task_node: TaskNode, context: TaskContext) -> TaskInstance:  # noqa: D102
        return TaskInstance(
            uid=context.run_uid,  # 这一步没有实际节点, uid与运行实例的uuid一致
            instance_uid=context.run_uid,
            owner_id=context.owner.id,
            name='summary',
            status=Status.WAITING,
        )

    def create_celery_task(self, task_node: TaskNode, context: TaskContext) -> Signature:  # noqa: D102
        summary_prompt = task_node.data.get('summary_prompt') if task_node.data else None
        return upstream_summary_task.s(
            base_dir=context.output_base,
            summary_prompt=summary_prompt,
        ).set(task_id=str(context.run_uid))
