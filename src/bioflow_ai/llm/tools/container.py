"""Agent shell tools, workspace mounts and workflow evidence source resolution."""

import asyncio
import posixpath
import re
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from anyio import Path as AsyncPath
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

import bioflow_ai.crud.project_file_mapping as project_file_mapping_crud
import bioflow_ai.crud.resource.sample as sample_crud
from bioflow_ai.core.config import get_settings
from bioflow_ai.infrastructure.containers import (
    ContainerCommandTimeoutError,
    ContainerMount,
    run_command_async,
    run_command_in_container,
)
from bioflow_ai.infrastructure.containers.mount_paths import (
    PROJECT_DATA_ROOTS,
    build_project_data_mount_plan,
)
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.llm.context import CodingContext, ProjectContext, TestRunContext, WorkflowAssistantContext
from bioflow_ai.llm.states import ProjectPlannerState
from bioflow_ai.models.workflow import NodeRun, WorkflowRun
from bioflow_ai.schemas.tool_config import ImageConfig
from bioflow_ai.services.code.workspace import CODE_IMAGES, build_workspace_mounts
from bioflow_ai.utils.llm.truncation import truncate_output

DEFAULT_LINUX_IMAGE = ImageConfig.from_str('docker.io/aye1032/biokit-utils:1.2.0')
INSPECTION_IMAGE = ImageConfig.from_str('docker.io/aye1032/biokit-seq:1.1.0')
EVIDENCE_MOUNT_PATH = '/app/data/input'
EVIDENCE_WORKDIR = '/app/data'
PROJECT_DATA_DIR = '/app/data'
PROJECT_WORK_DIR = '/work'


def _resolve_timeout(timeout: int | None, *, default: int, maximum: int) -> tuple[int, str | None]:
    if timeout is None:
        return default, None
    minimum = get_settings().agent_command.MIN_TIMEOUT
    if not (minimum <= timeout <= maximum):
        return default, f'Invalid timeout {timeout}s: must be between {minimum} and {maximum} seconds.'
    return timeout, None


def _format_timeout_notice(
    *,
    timeout: int,
    maximum: int,
    partial_output: str,
    mode: Literal['execution', 'inspection'],
    image_name: str | None = None,
) -> str:
    prefix = f'[Image: {image_name}]\n' if image_name else ''
    header = (
        f'{prefix}Command was intentionally terminated because it exceeded the {timeout}s timeout '
        '(this is a proactive timeout safeguard, not an unexpected crash).'
    )
    if mode == 'execution':
        guidance = (
            'Before retrying, consider:\n'
            '1. Downsampling or subsetting large input files (for example, creating a smaller test subset in the writable directory) so verification finishes quickly.\n'
            '2. Adding constraining or resource parameters (such as thread count, memory limits, or lighter modes).\n'
            f'3. Last resort only: if you are certain the tool inherently requires longer (for example, loading a large database like Kraken2), retry with a larger `timeout` (up to {maximum}s).'
        )
    else:
        guidance = (
            'Before retrying, consider:\n'
            '1. Narrowing the command or sampling the target instead of scanning entire large files (for example, using `head`, `tail`, `seqkit head`, `rg -m`, or limiting `find` depth).\n'
            f'2. Last resort only: if you are certain the inspection inherently requires longer, retry with a larger `timeout` (up to {maximum}s).'
        )
    sections = [header, guidance]
    if clean_output := partial_output.strip():
        sections.append(f'Partial output before termination:\n{truncate_output(clean_output)}')
    return '\n\n'.join(sections)


def _validate_command(command: str, workdir: str | None, tool_call_id: str | None) -> Command | None:
    if re.search(r'\brm\b', command) and (workdir and '/data/output' not in workdir) and '/data/output' not in command:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='It is forbidden to use `rm` command outside of the /data/output path!',
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    if re.search(r'\bcd\b', command):
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='The `cd` command is prohibited. To specify the working directory, please use the `workdir` parameter.',
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    return None


def _resolve_image_ref(
    image_ref: str | None,
    runtime: ToolRuntime[TestRunContext],
) -> tuple[ImageConfig, Command | None]:
    """Resolve an exact generated image reference to its container configuration."""
    if not image_ref:
        return DEFAULT_LINUX_IMAGE, None

    from bioflow_ai.services.tools import catalog as tool_catalog

    with Session(engine) as session:
        tool_image = tool_catalog.resolve_image_reference(session, image_ref)

    if tool_image is None:
        return (
            DEFAULT_LINUX_IMAGE,
            Command(
                update={
                    'messages': [
                        ToolMessage(
                            content=f'No local image has the reference `{image_ref}`. Use `search_local_image` and pass the exact `<image_ref>` from its results.',
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            ),
        )

    return tool_image.image, None


def _resolve_extra_mounts(
    mounts: list[ContainerMount], context: TestRunContext, recorded_mounts: list[ContainerMount]
) -> list[ContainerMount]:
    """Resolve fixture aliases and keep every source outside the trial read-only."""
    from bioflow_ai.llm.tools.test_run_report import resolve_test_run_input_path, resolve_test_run_path

    trial_root = resolve_test_run_path('/data/output', context.test_run_id)
    managed_mounts = ('/data/testdata', '/data/output', *PROJECT_DATA_ROOTS)
    resolved: dict[str, ContainerMount] = {}
    for mount in mounts:
        target = PurePosixPath(mount.target)
        if not target.is_absolute() or '..' in target.parts:
            message = f'The container mount target must be absolute without parent traversal: {mount.target}'
            raise ValueError(message)
        if any(PurePosixPath(root).is_relative_to(target) for root in managed_mounts):
            message = f'Mount {mount.target} would replace a default mount. Choose another target or a subdirectory.'
            raise ValueError(message)
        source = resolve_test_run_input_path(
            mount.source,
            context.test_run_id,
            context.project_id,
            [*recorded_mounts, *resolved.values()],
        ).resolve(strict=True)
        if not (source.is_file() or source.is_dir()):
            message = f'Mount source must be a file or directory: {source}'
            raise ValueError(message)
        normalized = mount.model_copy(
            update={
                'source': str(source),
                'target': str(target),
                'read_only': mount.read_only or not source.is_relative_to(trial_root),
            }
        )
        if str(target) in resolved and resolved[str(target)] != normalized:
            message = f'Multiple sources specified for mount target {target}'
            raise ValueError(message)
        resolved[str(target)] = normalized
    return list(resolved.values())


async def _validate_extra_mounts(
    mounts: list[ContainerMount] | None,
    tool_call_id: str | None,
    context: TestRunContext,
    recorded_mounts: list[ContainerMount] | None = None,
) -> tuple[list[ContainerMount], Command | None]:
    try:
        return await asyncio.to_thread(_resolve_extra_mounts, mounts or [], context, recorded_mounts or []), None
    except (ValueError, OSError) as error:
        return [], Command(
            update={'messages': [ToolMessage(content=str(error), tool_call_id=tool_call_id, status='error')]}
        )


@tool('run_command', parse_docstring=True)
async def advance_run_command_tool(
    command: str,
    runtime: ToolRuntime[TestRunContext],
    image_ref: str | None = None,
    workdir: str | None = '/data/output',
    mounts: list[ContainerMount] | None = None,
    timeout: int | None = None,
) -> Command:
    """在容器环境中运行指定 bash 命令或生物信息学命令行工具并返回执行结果.

    使用默认的较短超时时间; 超时会主动终止命令并返回提示. 仅在确定工具本身确实需要更长耗时(如加载 Kraken2 等大型数据库)时作为最后手段传入更大的 `timeout`.

    Args:
        command: 要运行的Linux命令行指令
        image_ref: `search_local_image` 返回的镜像 `<image_ref>`; 若不指定则使用默认的Linux基础镜像
        workdir: 指定命令运行的工作目录
        mounts: 除默认测试数据和输出目录外, **额外的**数据挂载
        timeout: 可选超时时间(秒). 仅作为最后手段用于确认需要长耗时的工具, 优先考虑对数据下采样或调整线程/内存等限制参数.
    """
    if invalid_command := _validate_command(command, workdir, runtime.tool_call_id):
        return invalid_command

    timeout_cfg = get_settings().agent_command
    resolved_timeout, timeout_error = _resolve_timeout(
        timeout,
        default=timeout_cfg.TRIAL_DEFAULT_TIMEOUT,
        maximum=timeout_cfg.TRIAL_MAX_TIMEOUT,
    )
    if timeout_error is not None:
        return Command(update={'messages': [ToolMessage(content=timeout_error, tool_call_id=runtime.tool_call_id)]})

    # 镜像解析
    image, invalid_image = _resolve_image_ref(image_ref, runtime)
    if invalid_image is not None:
        return invalid_image

    # 固定挂载
    base_path = await (await AsyncPath(get_settings().persistent.storage.STORAGE_PATH).expanduser()).resolve()
    output_path = base_path / 'testrun' / runtime.context.test_run_id
    await output_path.mkdir(parents=True, exist_ok=True)

    testdata_path = await (await AsyncPath(get_settings().persistent.storage.TEST_DATA_PATH).expanduser()).resolve()
    project_mounts = []
    if runtime.context.project_id is not None:
        project_mounts = list(build_project_data_mount_plan(runtime.context.project_id).mounts)

    extra_mounts, invalid_mount = await _validate_extra_mounts(
        mounts,
        runtime.tool_call_id,
        runtime.context,
        [ContainerMount.model_validate(mount) for mount in runtime.state.get('test_run_mounts', [])],
    )
    if invalid_mount is not None:
        return invalid_mount

    all_mounts = [
        ContainerMount(target='/data/testdata', source=str(testdata_path), read_only=True),
        ContainerMount(target='/data/output', source=str(output_path), read_only=False),
        *project_mounts,
        *extra_mounts,
    ]

    image_name = image_ref or 'linux_default'
    try:
        code, output = await run_command_async(image, all_mounts, command, workdir, timeout=resolved_timeout)
    except ContainerCommandTimeoutError as exc:
        timeout_msg = _format_timeout_notice(
            timeout=exc.timeout,
            maximum=timeout_cfg.TRIAL_MAX_TIMEOUT,
            partial_output=exc.output,
            mode='execution',
            image_name=image_name,
        )
        return Command(update={'messages': [ToolMessage(content=timeout_msg, tool_call_id=runtime.tool_call_id)]})

    status = 'successfully' if code == 0 else 'failed'

    if output.strip():
        if not any(k in command for k in ('about.md', 'DESCRIPTION.md')):
            output = truncate_output(output)
        return_msg = (
            f'[Image: {image_name}]\n{output}'
            if code == 0
            else f'[Image: {image_name}]\nCommand ran {status}.\n\n......\n{output}'
        )
    else:
        return_msg = f'[Image: {image_name}]\nCommand ran {status}. No other output.'

    update: dict[str, Any] = {
        'messages': [
            ToolMessage(
                content=return_msg,
                tool_call_id=runtime.tool_call_id,
            )
        ],
    }
    if code == 0 and extra_mounts:
        update['test_run_mounts'] = extra_mounts
    return Command(update=update)


@tool('run_command', parse_docstring=True)
async def coding_run_command(
    command: str,
    runtime: ToolRuntime[CodingContext],
    workdir: str = '/workspace',
    timeout: int | None = None,
) -> str:
    """Run shell commands in container.

    The authoring workspace is mounted read-write at /workspace. Commands can
    read or modify files and run checks. Each CODE language uses the same image
    for authoring checks and workflow execution. Commands use a short default
    timeout and are terminated if exceeded; only raise `timeout` as a last resort
    when you are certain the command legitimately needs longer.

    Args:
        command: Shell command to run; narrow large outputs.
        workdir: Working directory inside the container.
        timeout: Optional timeout in seconds. Last resort only; prefer downsampling or constraining parameters first.
    """
    timeout_cfg = get_settings().agent_command
    resolved_timeout, timeout_error = _resolve_timeout(
        timeout,
        default=timeout_cfg.CODING_DEFAULT_TIMEOUT,
        maximum=timeout_cfg.CODING_MAX_TIMEOUT,
    )
    if timeout_error is not None:
        return timeout_error

    mounts = build_workspace_mounts(runtime.context.workspace_path, runtime.context.node_type)
    image = CODE_IMAGES[runtime.context.node_type]
    try:
        code, output = await run_command_async(image, mounts, command, workdir, timeout=resolved_timeout)
    except ContainerCommandTimeoutError as exc:
        return _format_timeout_notice(
            timeout=exc.timeout,
            maximum=timeout_cfg.CODING_MAX_TIMEOUT,
            partial_output=exc.output,
            mode='execution',
        )

    msgs = []
    if code != 0:
        msgs.append(f'Exit code: {code}')
    if output.strip():
        if 'AGENTS.md' not in command:
            output = truncate_output(output)
        msgs.append(output)
    else:
        msgs.append('(no output)')
    return '\n'.join(msgs)


class ResolvedEvidence(BaseModel):
    """A validated host resource that may be mounted into the evidence container."""

    model_config = ConfigDict(frozen=True)

    source: str
    path: Path
    kind: Literal['file', 'directory']


def _validated_path(raw_path: str | None, source: str) -> tuple[Path, Literal['file', 'directory']]:
    if not raw_path:
        message = f'Evidence source `{source}` has no filesystem path.'
        raise ValueError(message)
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        message = f'Evidence source `{source}` no longer exists.'
        raise ValueError(message) from error
    if path.is_file():
        return path, 'file'
    if path.is_dir():
        return path, 'directory'
    message = f'Evidence source `{source}` is not a regular file or directory.'
    raise ValueError(message)


def resolve_evidence_source(source: str, context: WorkflowAssistantContext) -> ResolvedEvidence:
    """Resolve one logical evidence reference without accepting arbitrary host paths."""
    with Session(engine) as session:
        run = resolve_workflow_run(session, context, None)

        if source == 'run:current':
            path, kind = _validated_path(run.base_dir, source)
            if kind != 'directory':
                message = 'The current run path is not a directory.'
                raise ValueError(message)
            return ResolvedEvidence(source=source, path=path, kind=kind)

        if source.startswith('run:child:'):
            child = resolve_workflow_run(session, context, source.removeprefix('run:child:'))
            path, kind = _validated_path(child.base_dir, source)
            if kind != 'directory':
                message = 'The child run path is not a directory.'
                raise ValueError(message)
            return ResolvedEvidence(source=source, path=path, kind=kind)

        if source.startswith('sample:'):
            tag = source.removeprefix('sample:')
            if not tag:
                message = 'A sample source must include a tag.'
                raise ValueError(message)
            if run.sample_uid is None:
                message = 'The current workflow execution is project-level and has no bound sample.'
                raise ValueError(message)

            sample = sample_crud.get_with_files(session, run.sample_uid)
            if sample is None:
                message = 'The sample bound to the current run no longer exists.'
                raise ValueError(message)

            sample_file = next((item for item in sample.sample_files if item.tag == tag), None)
            if sample_file is None:
                message = f'Sample reference `{source}` not found.'
                raise ValueError(message)
            path, kind = _validated_path(sample_file.file_path, source)
            return ResolvedEvidence(source=source, path=path, kind=kind)

        if source.startswith('proj:'):
            keyword = source.removeprefix('proj:')
            if not keyword:
                message = 'A project source must include a keyword.'
                raise ValueError(message)
            mapping = project_file_mapping_crud.get_by_keyword(session, context.project_id, keyword)
            if mapping is None:
                message = f'Project reference `{source}` not found.'
                raise ValueError(message)

            path, kind = _validated_path(mapping.file_path, source)
            return ResolvedEvidence(source=source, path=path, kind=kind)

    message = 'source must be `run:current`, `run:child:<uid>`, `sample:<tag>`, or `proj:<keyword>`.'
    raise ValueError(message)


def _descendant_run(session: Session, root: WorkflowRun, uid: uuid.UUID) -> WorkflowRun | None:
    """Only expose runs linked by ForEach parent nodes of the root run."""
    child = session.get(WorkflowRun, uid)
    current = child
    for _ in range(33):
        if current is None or current.owner_id != root.owner_id or current.project_id != root.project_id:
            return None
        if current.uid == root.uid:
            return child
        parent_node = session.get(NodeRun, current.parent_node_uid)
        if parent_node is None:
            return None
        current = session.get(WorkflowRun, parent_node.run_uid)
    return None


def resolve_workflow_run(
    session: Session, context: WorkflowAssistantContext, child_run_uid: str | None = None
) -> WorkflowRun:
    """Select the current run or a verified descendant within the source generation."""
    run_uid = uuid.UUID(context.source_run_uid)
    root = session.get(WorkflowRun, run_uid)
    if root is None:
        msg = 'Run instance not found'
        raise ValueError(msg)
    if not child_run_uid:
        return root
    try:
        child = _descendant_run(session, root, uuid.UUID(child_run_uid))
    except ValueError as error:
        message = 'Invalid child run UID.'
        raise ValueError(message) from error
    if child is None:
        message = 'Child run does not belong to this source run generation.'
        raise ValueError(message)
    return child


@tool('run_command', parse_docstring=True)
async def run_evidence_command(
    source: str,
    command: str,
    runtime: ToolRuntime[WorkflowAssistantContext],
    timeout: int | None = None,
) -> str:
    """Run a shell inspection against one read-only workflow evidence source.

    The source is mounted at ``/app/data/input``. It may be a file or a directory, so
    inspect it with ``file /app/data/input`` when the type is not already known.
    The command is passed directly to the container executor. Long output is shortened
    in the returned preview, preserving its beginning and end. Commands use a short
    default timeout and are terminated if exceeded; only raise ``timeout`` as a last
    resort after trying narrower or sampled inspection commands.

    Args:
        source: ``run:current``, ``run:child:<uid>``, ``sample:<tag>``, or ``proj:<keyword>``.
        command: Shell inspection command using ``/app/data/input`` as the evidence path.
        timeout: Optional timeout in seconds. Last resort only; prefer sampling or narrowing the inspection command first.
    """
    timeout_cfg = get_settings().agent_command
    resolved_timeout, timeout_error = _resolve_timeout(
        timeout,
        default=timeout_cfg.INSPECTION_DEFAULT_TIMEOUT,
        maximum=timeout_cfg.INSPECTION_MAX_TIMEOUT,
    )
    if timeout_error is not None:
        return timeout_error

    try:
        evidence = await asyncio.to_thread(resolve_evidence_source, source, runtime.context)
    except ValueError as error:
        return f'Error: {error}'

    mount = ContainerMount(source=str(evidence.path), target=EVIDENCE_MOUNT_PATH, read_only=True)
    try:
        code, output = await run_command_async(
            INSPECTION_IMAGE, [mount], command, EVIDENCE_WORKDIR, timeout=resolved_timeout
        )
    except ContainerCommandTimeoutError as exc:
        return _format_timeout_notice(
            timeout=exc.timeout,
            maximum=timeout_cfg.INSPECTION_MAX_TIMEOUT,
            partial_output=exc.output,
            mode='inspection',
        )
    except Exception as error:
        return f'Error executing command: {type(error).__name__}: {error}'

    msgs = []
    if code != 0:
        msgs.append(f'Exit code: {code}')
    if output.strip():
        if not any(k in command for k in ('about.md', 'DESCRIPTION.md')):
            output = truncate_output(output)
        msgs.append(output)
    else:
        msgs.append('(no output)')
    return '\n'.join(msgs)


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(f'{root}/')


def _resolve_container_path(path: str, cwd: str) -> str:
    """Resolve a container path against ``cwd`` and normalize ``.``/``..`` without touching the host."""
    if '\\' in path or '\x00' in path:
        msg = f'Invalid path: {path!r}'
        raise ValueError(msg)
    return posixpath.normpath(posixpath.join(cwd, path))


def resolve_project_cwd(workdir: str | None, default_dir: str) -> str:
    """Resolve an explicit per-call working dir; it must lie inside /work or /app/data."""
    if not workdir:
        return default_dir
    cwd = _resolve_container_path(workdir, default_dir)
    if not (_under(cwd, PROJECT_WORK_DIR) or _under(cwd, PROJECT_DATA_DIR)):
        msg = f'Working dir must be inside {PROJECT_WORK_DIR} or {PROJECT_DATA_DIR}: {workdir}'
        raise ValueError(msg)
    return cwd


@tool('run_command', parse_docstring=True)
async def planner_run_command(
    command: str,
    runtime: ToolRuntime[ProjectContext, ProjectPlannerState],
    workdir: str | None = None,
    timeout: int | None = None,
) -> str:
    """Run a read-only shell command in a container.

    /app/data (the data dir set by `set_base_dir`) and /work (the project dir) are both mounted read-only.
    Relative paths resolve against the current default dir unless `workdir` is given.
    Available commands include seqkit, fd, rg, jq, yq, mlr, awk, grep, find, head and file.
    Commands use a short default timeout and are terminated if exceeded; only raise `timeout`
    as a last resort after trying narrower or sampled inspection commands.

    Args:
        command: Shell command to run in the container.
        workdir: Working dir for this call only, inside /work or /app/data. Defaults to the current default dir.
        timeout: Optional timeout in seconds. Last resort only; prefer sampling or narrowing the command first.
    """
    timeout_cfg = get_settings().agent_command
    resolved_timeout, timeout_error = _resolve_timeout(
        timeout,
        default=timeout_cfg.INSPECTION_DEFAULT_TIMEOUT,
        maximum=timeout_cfg.INSPECTION_MAX_TIMEOUT,
    )
    if timeout_error is not None:
        return timeout_error

    try:
        cwd = resolve_project_cwd(workdir, runtime.state.get('default_dir', PROJECT_WORK_DIR))
    except ValueError as exc:
        return str(exc)

    project_dir = Path(runtime.context.workdir)
    await asyncio.to_thread(project_dir.mkdir, parents=True, exist_ok=True)
    mounts = [ContainerMount(source=str(project_dir), target=PROJECT_WORK_DIR, read_only=True)]
    if base_dir := runtime.state.get('base_path'):
        if not await asyncio.to_thread(Path(base_dir).is_dir):
            return (
                f'The data dir {base_dir} no longer exists. Call `set_base_dir` with the correct path, '
                'or ask the user to confirm it.'
            )
        mounts.append(ContainerMount(source=base_dir, target=PROJECT_DATA_DIR, read_only=True))
    elif _under(cwd, PROJECT_DATA_DIR):
        return f'{PROJECT_DATA_DIR} is not mounted. Call `set_base_dir` with the data path first.'

    try:
        code, result = await run_command_async(INSPECTION_IMAGE, mounts, command, cwd, timeout=resolved_timeout)
    except ContainerCommandTimeoutError as exc:
        return _format_timeout_notice(
            timeout=exc.timeout,
            maximum=timeout_cfg.INSPECTION_MAX_TIMEOUT,
            partial_output=exc.output,
            mode='inspection',
        )

    run_status = 'successfully' if code == 0 else 'failed'
    if clean_out := result.strip():
        return (
            _truncate_project_output(clean_out)
            if code == 0
            else f'Command ran {run_status}.\n\n......\n{_truncate_project_output(clean_out)}'
        )
    return f'Command ran {run_status}. No other output.'


def _truncate_project_output(output: str) -> str:
    if len(output) <= 3000:  # noqa: PLR2004
        return output
    return f'{output[:1500]}\n...\n{output[-500:]}'


def _truncate_report_output(command: str, output: str) -> str:
    if any(k in command for k in ('find', 'grep', 'help', 'summary.md')) or len(output) <= 3000:  # noqa: PLR2004
        return output
    return f'{output[:1500]}\n...\n{output[-500:]}'


@tool('run_command')
async def report_run_command(command: str, runtime: ToolRuntime, workdir: str | None = None) -> str:
    """Execute shell commands within the container."""
    runtime.stream_writer(f'run: {command[:100]}')
    container_id = runtime.state.get('container_id', '')
    code, out = await asyncio.to_thread(run_command_in_container, container_id, command, workdir)

    run_status = 'successfully' if code == 0 else 'failed'
    if out.strip():
        return_msg = (
            _truncate_report_output(command, out)
            if code == 0
            else f'Command ran {run_status}.\n\n{_truncate_report_output(command, out)}'
        )
    else:
        return_msg = f'Command ran {run_status}. No other output.'

    return return_msg
