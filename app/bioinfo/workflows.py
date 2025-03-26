from loguru import logger
from prefect import flow
from prefect.context import get_run_context

from app.bioinfo.tasks.assembly import spades
from app.bioinfo.tasks.qc import fastp


@flow(log_prints=True)
async def main_flow(r1: str, r2: str):
    ctx = get_run_context()
    logger.info(ctx.flow_run.id)
    clean_result = await fastp(r1, r2, output_dir='test/clean')

    spades_result = await spades(
        clean_result.r1, clean_result.r2, 'test/assembly', mode='sc', thread=64
    )
