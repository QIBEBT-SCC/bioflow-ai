import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import markdown
import palitra
from celery import Task
from jinja2 import Environment, FileSystemLoader
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.models.tools import ImageConfig
from app.models.workflow import ToolOutput
from app.schemas.react_flow import TaskInput
from app.tasks.base.base_task import BaseBioTask, get_tool_output
from app.utils.container import ContainerMount, create_container, remove_container
from llm.agents.reporter.summary_analyse import summary_analyse_agent
from llm.states import FolderContext
from worker import celery_app


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def downstream_summary_task(
    self: Task,  # noqa: ARG001
    result_dict: Optional[dict[str, Any]] = None,  # noqa: ARG001
    *,
    inputs: dict[str, TaskInput | list[TaskInput]],
    prompt: str,
    output_dir: str,
) -> ToolOutput:
    """将下游分析文件汇总, 同时生成编码prompt."""
    input_files = inputs.get('input_files')
    kwargs = input_files if isinstance(input_files, list) else [input_files]

    output_folder = Path(output_dir)
    output_folder.unlink(missing_ok=True)
    output_folder.mkdir(parents=True)

    for kwarg in kwargs:
        origin_file = Path(get_tool_output(kwarg.source_uid, kwarg.source_k)[0])
        if origin_file.exists():
            new_file = output_folder / origin_file.name
            if new_file.is_symlink() or new_file.is_file():
                new_file.unlink()
            elif new_file.is_dir():
                shutil.rmtree(new_file)
            new_file.symlink_to(origin_file)
        else:
            logger.warning(f'file {origin_file} did not exist!')

    # 下游代码 vibe coding 脚本
    claude_command = f'claude --dangerously-skip-permissions --output-format=stream-json --verbose -p "{prompt}"'
    claude_bash = output_folder.parent / 'run_downstream.sh'
    claude_bash.unlink(missing_ok=True)
    claude_bash.write_text(claude_command)

    return ToolOutput(result={})


async def _get_summary(agent_input: dict[str, Any], config: RunnableConfig, context: FolderContext) -> None:
    async for _, _, data in summary_analyse_agent.astream(
        agent_input,  # type: ignore
        config,
        context=context,  # type: ignore
        stream_mode=['custom'],
        subgraphs=True,
    ):
        logger.debug(data)


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def upstream_summary_task(
    self: Task,  # noqa: ARG001
    result_dict: Optional[dict[str, Any]] = None,  # noqa: ARG001
    *,
    base_dir: str,
) -> ToolOutput:
    """生成上游分析总结报告."""
    image = ImageConfig.from_str('docker.io/aye1032/biokit-utils:1.1.0')
    rawdata_path = Path(base_dir)
    summary_path = rawdata_path / 'summary'

    if summary_path.exists():
        shutil.rmtree(summary_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)

        file_mount = [
            ContainerMount(source=f'{rawdata_path.absolute()}', target='/data/raw', read_only=True),
            ContainerMount(source=f'{work_dir.absolute()}', target='/data/output', read_only=False),
        ]
        container_id = create_container(image, file_mount)

        agent_input = {
            'messages': [HumanMessage(content='请阅读当前目录下的分析项目, 给出分析报告')],
            'container_id': container_id,
        }
        config: RunnableConfig = {'recursion_limit': sys.maxsize}
        context = FolderContext(
            output_path=temp_dir,
            user_id=2,
        )
        palitra.run(_get_summary(agent_input, config, context))

        shutil.copytree(work_dir, summary_path, dirs_exist_ok=True)

    remove_container(container_id)

    template_dir = Path(__file__).parent / 'templates'
    output_path = summary_path / 'summary.html'
    md_content = (summary_path / 'summary.md').read_text(encoding='utf-8')
    html = _md_to_html(md_content, template_dir)
    output_path.write_text(html, encoding='utf-8')
    (summary_path / 'summary.md').unlink(missing_ok=True)

    return ToolOutput(result={})


def _md_to_html(md_content: str, template_dir: str | Path, template_name: str = 'report.html.j2') -> str:
    # Markdown -> HTML 片段
    html_body = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'toc', 'attr_list'],
    )

    # 提取标题 (第一个 h1)
    title = '分析报告'
    for line in md_content.splitlines():
        if line.startswith('# '):
            title = line.lstrip('# ').strip()
            break

    # 加载模板并渲染
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
    template = env.get_template(template_name)
    return template.render(title=title, content=html_body)
