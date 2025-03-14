from prefect import flow

from app.bioinfo.tasks.assembly import bb_norm, spades
from app.bioinfo.tasks.qc import fastp


@flow(log_prints=True)
async def main_flow():
    clean_result = await fastp(
        'test/raw/M3_1.fq.gz', 'test/raw/M3_2.fq.gz', output_dir='test/clean'
    )

    norm_result = await bb_norm(clean_result.r1, clean_result.r2, 'test/assembly')
    spades_result = await spades(
        norm_result.r1, norm_result.r2, 'test/assembly', mode='sc', thread=64
    )
