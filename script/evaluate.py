import asyncio
import sys

import click
from loguru import logger

from llm.utils.render import print_stream_chunk


async def run_eval(task_file: str, paper_file: str | None, thread_id: str, *, drop_old: bool) -> None:
    """运行bioflow ai 测试."""
    import base64
    import os
    from pathlib import Path

    import rich
    from langchain_core.messages import HumanMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.redis import AsyncRedisSaver
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
    from phoenix.otel import register
    from rich.console import Console

    from app.core.config import get_settings
    from llm.agents.reproduction.design_workflow import WorkflowDesigner
    from llm.states import DesignContext
    from llm.utils.multimodal import filter_multimedia_message

    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_PROJECT'] = 'bioflow-ai-eval'

    # configure the Phoenix tracer
    tracer_provider = register(  # noqa: F841
        endpoint=get_settings().develop.PHOENIX_ENDPOINT,
        project_name='bioflow-ai-eval',
        batch=True,
        auto_instrument=True,
        set_global_tracer_provider=False,
    )

    task_path = Path(task_file)
    if not task_path.exists():
        logger.error('Task file not found!')
        sys.exit(0)

    if paper_file:
        paper_path = Path(paper_file)
        if not paper_path.exists():
            logger.error('Paper file not found!')
            sys.exit(0)

        base64_encoded = base64.b64encode(paper_path.read_bytes()).decode('utf-8')

        init_human_message = HumanMessage(
            content=[
                {
                    'type': 'text',
                    'text': task_path.read_text(encoding='utf-8'),
                },
                {
                    'type': 'file',
                    'base64': base64_encoded,
                    'mime_type': 'application/pdf',
                    'filename': paper_path.name,
                },
            ]
        )
        human_input = await filter_multimedia_message(init_human_message)
    else:
        human_input = HumanMessage(
            content=[
                {
                    'type': 'text',
                    'text': task_path.read_text(encoding='utf-8'),
                },
                {
                    'type': 'text',
                    'text': 'The above is all the information and requirements I can provide. The rest is entirely up to you. Do not ask me for any additional information; I have nothing more to tell you.',
                },
            ]
        )

    console = Console()

    async with AsyncRedisSaver.from_conn_string(
        get_settings().persistent.cache.LLM_CACHE.unicode_string()
    ) as checkpoint:
        checkpoint.serde = JsonPlusRedisSerializer(
            allowed_msgpack_modules=[
                ('llm.utils.middleware.todo', 'Todo'),
                ('llm.states', 'BuildEdge'),
                ('llm.states', 'BuildNode'),
                ('llm.utils.uid_mapper', 'BidirectionalMapper'),
            ]
        )
        await checkpoint.setup()

        agent = WorkflowDesigner(checkpoint).build()

        if drop_old:
            await checkpoint.adelete_thread(thread_id)

        config = RunnableConfig(configurable={'thread_id': thread_id}, recursion_limit=sys.maxsize)
        agent_input = {'messages': [human_input]}

        async for _, stream_mode, chunk in agent.astream(
            agent_input,  # type: ignore
            config,
            context=DesignContext(user_id=2, auto_approve=True),  # type: ignore
            stream_mode=['updates', 'custom'],
            subgraphs=True,
        ):
            print_stream_chunk(console, stream_mode, chunk)

        state = await agent.aget_state(config)
        # 有中断
        if state.tasks and any(hasattr(t, 'interrupts') and t.interrupts for t in state.tasks):
            for task in state.tasks:
                if hasattr(task, 'interrupts'):
                    for interrupt in task.interrupts:
                        rich.print(interrupt)
                        return

        workflow_uid = state.values.get('workflow_uids', [])[0]
        logger.success(f'工作流已生成: {workflow_uid}')
        return


async def continue_run(message: str, thread_id: str) -> None:
    """响应中断."""
    import os

    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.redis import AsyncRedisSaver
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
    from langgraph.types import Command

    from app.core.config import get_settings
    from llm.agents.reproduction.design_workflow import WorkflowDesigner
    from llm.states import DesignContext

    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_PROJECT'] = 'bioflow-ai-eval'

    async with AsyncRedisSaver.from_conn_string(
        get_settings().persistent.cache.LLM_CACHE.unicode_string()
    ) as checkpoint:
        checkpoint.serde = JsonPlusRedisSerializer(
            allowed_msgpack_modules=[
                ('llm.utils.middleware.todo', 'Todo'),
                ('llm.states', 'BuildEdge'),
                ('llm.states', 'BuildNode'),
                ('llm.utils.uid_mapper', 'BidirectionalMapper'),
            ]
        )
        await checkpoint.setup()

        agent = WorkflowDesigner(checkpoint).build(visualize=True)

        config = RunnableConfig(configurable={'thread_id': thread_id}, recursion_limit=sys.maxsize)
        agent_input = Command(resume=message)

        async for _, stream_mode, chunk in agent.astream(
            agent_input,
            config,
            context=DesignContext(user_id=2, auto_approve=True),  # type: ignore
            stream_mode=['updates', 'custom'],
            subgraphs=True,
        ):
            match stream_mode:
                case 'custom':
                    logger.info(chunk)
                case 'updates':
                    logger.debug(chunk)
                case _:
                    pass

        agent_result = await agent.aget_state(config)
        # 有中断
        if agent_result.tasks and any(hasattr(t, 'interrupts') and t.interrupts for t in agent_result.tasks):
            pass


@click.group()
def cli() -> None:
    """Command for agent evaluate."""


@cli.command()
@click.argument('task_file', type=click.Path(exists=True))
@click.option('--paper', '-p', 'paper_file', default=None, type=click.Path(exists=True), help='PDF paper file path.')
@click.option('--thread-id', '-t', help='Checkpoint thread ID.')
@click.option('--drop-old', is_flag=True, default=False, help='Drop existing checkpoint before running.')
def run_cmd(task_file: str, paper_file: str | None, thread_id: str, drop_old: bool) -> None:  # noqa: FBT001
    """Run the workflow design evaluation on TASK_FILE."""
    asyncio.run(run_eval(task_file, paper_file, thread_id, drop_old=drop_old))


if __name__ == '__main__':
    cli()
