from loguru import logger
from prefect import flow

from app.bioinfo.tasks.assembly import rename, bb_norm, spades
from app.bioinfo.tasks.qc import fastp
from app.utils.logger import init_logging


@flow(log_prints=True)
def main_flow() -> None:
    clean_result = fastp('test/raw/M3_1.fq.gz', 'test/raw/M3_2.fq.gz', output_dir='test/clean')
    logger.info(clean_result)
    rename_1 = rename(clean_result.r1, 'test/assembly')
    rename_2 = rename(clean_result.r2, 'test/assembly')

    norm_result = bb_norm(rename_1, rename_2, 'test/assembly')
    spades_result = spades(norm_result.r1, norm_result.r2, 'test/assembly', mode='sc', thread=64)


if __name__ == '__main__':
    init_logging()
    main_flow()
